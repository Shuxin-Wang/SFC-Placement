import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch_geometric.data import Data
from actor import StateNetworkActor, Seq2SeqActor, DecoderActor
from critic import StateNetworkCritic, LSTMCritic, DecoderCritic

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):    # state: (net_state, sfc_state)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class NCO(nn.Module):
    def __init__(self, cfg, env, sfc_generator, device='cpu'):
        super().__init__()
        self.node_state_dim, self.vnf_state_dim, _, _ = env.get_state_dim(sfc_generator)
        self.num_nodes = env.num_nodes
        self.episode = cfg.episode
        self.batch_size = cfg.batch_size
        self.max_sfc_length = cfg.max_sfc_length

        self.actor = Seq2SeqActor(self.node_state_dim, self.vnf_state_dim, self.num_nodes, self.max_sfc_length, hidden_dim=8, num_layers=2).to(device)
        self.critic = LSTMCritic(self.node_state_dim, self.vnf_state_dim, self.num_nodes, hidden_dim=8).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = ReplayBuffer(capacity=self.episode * self.batch_size)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.avg_episode_reward = 0
        self.avg_acceptance_ratio = 0
        self.avg_node_resource_utilization = 0

    def select_action(self, logits, exploration=True):
        if exploration:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        return action   # 1 * max_sfc_length

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        self.avg_acceptance_ratio = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list, source_dest_node_pairs, reliability_requirement_list = sfc_generator.get_sfc_states()
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                reliability_requirement = reliability_requirement_list[i]
                state = (net_state.to(self.device),
                         sfc_state.to(self.device),
                         source_dest_node_pair.to(self.device),
                         reliability_requirement.to(self.device))

                with torch.no_grad():
                    logits, probs = self.actor([state])

                action = self.select_action(logits, exploration=True)
                placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i] + reliability_requirement.tolist()
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_reliability_requirement = reliability_requirement_list[i + 1]
                    next_state = (next_net_state.to(self.device),
                                  next_sfc_state.to(self.device),
                                  next_source_dest_node_pair.to(self.device),
                                  next_reliability_requirement.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            self.avg_acceptance_ratio += env.sfc_placed_num
            env.clear()
        return self.avg_episode_reward / episode, self.avg_acceptance_ratio / sfc_generator.batch_size / episode

    def train(self, episode=1, discount=0.99, tau=0.005):
        batch_size = self.episode * self.batch_size

        self.actor_loss_list.clear()
        self.critic_loss_list.clear()

        for e in range(episode):
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(
                *self.replay_buffer.sample(batch_size))
            states = list(batch_states)  # state = [(net_state, sfc_state), (net_state, sfc_state)...]
            actions = torch.stack(batch_actions, dim=0).squeeze(1).to(
                self.device)  # action = torch.tensor((action), (action)...)) batch_size * max_sfc_length
            rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            next_states = list(batch_next_states)
            dones = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(self.device)

            logits, probs = self.actor(states)   # batch_size * max_sfc_length * node_num
            log_probs = F.log_softmax(logits, dim=-1)
            log_pi_action = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # get the log probs for the actions: batch_size * max_sfc_length
            log_pi_action = log_pi_action.sum(dim=-1, keepdim=True)

            with torch.no_grad():
                baseline = self.critic(states)

            advantage = rewards - baseline
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            actor_loss = -(advantage * log_pi_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_loss_list.append(actor_loss.item())

            values = self.critic(states)
            critic_loss = F.mse_loss(values, rewards)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_loss_list.append(critic_loss.item())

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs, reliability_requirement_list):
        num_sfc = len(sfc_list)
        env.clear()
        for i in range(num_sfc):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            reliability_requirement = reliability_requirement_list[i]
            state = (net_state.to(self.device),
                     sfc_state.to(self.device),
                     source_dest_node_pair.to(self.device),
                     reliability_requirement.to(self.device))

            with torch.no_grad():
                logits, probs = self.actor([state])

            action = self.select_action(logits, exploration=False)  # action_dim: batch_size * max_sfc_length * 1
            placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i] + reliability_requirement.tolist()
            env.step(sfc, placement)

class EnhancedNCO(nn.Module):
    def __init__(self, cfg, env, sfc_generator, device='cpu'):
        super().__init__()
        self.node_state_dim, self.vnf_state_dim, _, _ = env.get_state_dim(sfc_generator)
        self.num_nodes = env.num_nodes
        self.episode = cfg.episode
        self.batch_size = cfg.batch_size
        self.max_sfc_length = cfg.max_sfc_length

        self.actor = StateNetworkActor(self.node_state_dim, self.vnf_state_dim, self.num_nodes, self.max_sfc_length).to(device)
        self.critic = StateNetworkCritic(self.node_state_dim, self.vnf_state_dim, self.num_nodes, self.max_sfc_length, hidden_dim=64).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = ReplayBuffer(capacity=self.episode * self.batch_size)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.avg_episode_reward = 0
        self.avg_acceptance_ratio = 0
        self.avg_node_resource_utilization = 0

    def select_action(self, logits, exploration=True):
        if exploration:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        return action

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        self.avg_acceptance_ratio = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list, source_dest_node_pairs, reliability_requirement_list = sfc_generator.get_sfc_states()
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                reliability_requirement = reliability_requirement_list[i]
                state = (net_state.to(self.device),
                         sfc_state.to(self.device),
                         source_dest_node_pair.to(self.device),
                         reliability_requirement.to(self.device))

                with torch.no_grad():
                    logits, probs = self.actor([state])

                action = self.select_action(logits, exploration=True)
                placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist() # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i] + reliability_requirement.tolist()
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_reliability_requirement = reliability_requirement_list[i + 1]
                    next_state = (next_net_state.to(self.device),
                                  next_sfc_state.to(self.device),
                                  next_source_dest_node_pair.to(self.device),
                                  next_reliability_requirement.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            self.avg_acceptance_ratio += env.sfc_placed_num
            env.clear()
        return self.avg_episode_reward / episode, self.avg_acceptance_ratio / sfc_generator.batch_size / episode

    def train(self, episode=1, discount=0.99, tau=0.005):
        batch_size = self.episode * self.batch_size

        self.actor_loss_list.clear()
        self.critic_loss_list.clear()

        for e in range(episode):
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(
                *self.replay_buffer.sample(batch_size))
            states = list(batch_states)  # state = [(net_state, sfc_state), (net_state, sfc_state)...]
            actions = torch.stack(batch_actions, dim=0).squeeze(1).to(
                self.device)  # action = torch.tensor((action), (action)...)) batch_size * max_sfc_length
            rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            next_states = list(batch_next_states)
            dones = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(self.device)

            with torch.no_grad():
                next_values = self.critic(next_states)
                target_values = rewards + discount * next_values * (1 - dones)

            current_values = self.critic(states)
            critic_loss = F.smooth_l1_loss(target_values, current_values)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()
            self.critic_loss_list.append(critic_loss.item())

            logits, probs = self.actor(states)  # batch_size * max_sfc_length * node_num
            log_probs = F.log_softmax(logits, dim=-1)
            log_pi_action = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # get the log probs for the actions: batch_size * max_sfc_length
            log_pi_action = log_pi_action.sum(dim=-1, keepdim=True)

            with torch.no_grad():
                baseline = self.critic(states)
                advantage = target_values - baseline
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            actor_loss = -(advantage * log_pi_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()
            self.actor_loss_list.append(actor_loss.item())

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs, reliability_requirement_list):
        num_sfc = len(sfc_list)
        env.clear()
        for i in range(num_sfc):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            reliability_requirement = reliability_requirement_list[i]
            state = (net_state.to(self.device),
                     sfc_state.to(self.device),
                     source_dest_node_pair.to(self.device),
                     reliability_requirement.to(self.device))

            with torch.no_grad():
                logits, probs = self.actor([state])

            action = self.select_action(logits, exploration=False)  # action_dim: batch_size * max_sfc_length * 1
            placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i] + reliability_requirement.tolist()
            env.step(sfc, placement)

class PPO(nn.Module):
    def __init__(self, cfg, env, sfc_generator, device='cpu'):
        super().__init__()
        self.node_state_dim, self.vnf_state_dim, _, _ = env.get_state_dim(sfc_generator)
        self.num_nodes = env.num_nodes
        self.episode = cfg.episode
        self.batch_size = cfg.batch_size
        self.max_sfc_length = cfg.max_sfc_length

        self.actor = StateNetworkActor(self.node_state_dim, self.vnf_state_dim, self.num_nodes, self.max_sfc_length).to(device)
        self.critic = StateNetworkCritic(self.node_state_dim, self.vnf_state_dim, self.num_nodes, self.max_sfc_length, hidden_dim=64).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = ReplayBuffer(capacity=self.episode * self.batch_size)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.avg_episode_reward = 0
        self.avg_acceptance_ratio = 0
        self.avg_node_resource_utilization = 0

    def select_action(self, logits, exploration=True):
        if exploration:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        return action

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        self.avg_acceptance_ratio = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list, source_dest_node_pairs, reliability_requirement_list = sfc_generator.get_sfc_states()
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                reliability_requirement = reliability_requirement_list[i]
                state = (net_state.to(self.device),
                         sfc_state.to(self.device),
                         source_dest_node_pair.to(self.device),
                         reliability_requirement.to(self.device))

                with torch.no_grad():
                    logits, probs = self.actor([state])

                action = self.select_action(logits, exploration=True)
                placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist() # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i] + reliability_requirement.tolist()
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_reliability_requirement = reliability_requirement_list[i + 1]
                    next_state = (next_net_state.to(self.device),
                                  next_sfc_state.to(self.device),
                                  next_source_dest_node_pair.to(self.device),
                                  next_reliability_requirement.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            self.avg_acceptance_ratio += env.sfc_placed_num
            env.clear()
        return self.avg_episode_reward / episode, self.avg_acceptance_ratio / sfc_generator.batch_size / episode

    def train(self, episode=1, discount=0.99, clip_epsilon=0.2, ppo_epochs=4, gae_lambda=0.95):
        batch_size = self.episode * self.batch_size

        self.actor_loss_list.clear()
        self.critic_loss_list.clear()

        avg_actor_loss = 0
        avg_critic_loss = 0

        # GAE
        len_traj = self.batch_size
        num_traj = self.episode

        buffer_data = list(self.replay_buffer.buffer)

        all_states = []
        all_actions = []
        all_advantages = []
        all_targets = []

        for i in range(num_traj):
            traj = buffer_data[i * len_traj : (i + 1) * len_traj]
            states, actions, rewards, next_states, dones = zip(*traj)

            states = list(states)
            next_states = list(next_states)
            actions = torch.stack(actions, dim=0).squeeze(1).to(self.device)  # len_traj *  max_sfc_length
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                values = self.critic(states).squeeze(-1)    # len_traj
                next_values = self.critic(next_states).squeeze(-1)  # len_traj

            deltas = rewards + discount * next_values * (1 - dones) - values
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len_traj)):
                gae = deltas[t] + discount * gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            targets = advantages + values   # len_traj

            all_states.extend(states)
            all_actions.extend(actions)
            all_advantages.append(advantages)
            all_targets.append(targets)

        all_actions = torch.cat(all_actions, dim=0).view(batch_size, -1) # batch_size * max_sfc_length
        all_advantages = torch.cat(all_advantages, dim=0).unsqueeze(1)  # batch_size * 1
        all_targets = torch.cat(all_targets, dim=0).unsqueeze(1)    # batch_size * 1

        with torch.no_grad():
            old_logits, _ = self.actor(all_states)
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_log_pi_action = old_log_probs.gather(dim=-1, index=all_actions.unsqueeze(-1)).squeeze(-1)
            old_log_pi_action = old_log_pi_action.sum(dim=1, keepdim=True)

        for _ in range(ppo_epochs):
            logits, _ = self.actor(all_states)
            log_probs = F.log_softmax(logits, dim=-1)
            log_pi_action = log_probs.gather(dim=-1, index=all_actions.unsqueeze(-1)).squeeze(-1)
            log_pi_action = log_pi_action.sum(dim=1, keepdim=True)

            log_pi_diff = torch.clamp(log_pi_action - old_log_pi_action, -5, 5)
            ratio = torch.exp(log_pi_diff)

            v1 = ratio * all_advantages
            v2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * all_advantages
            actor_loss = -torch.min(v1, v2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()
            avg_actor_loss += actor_loss.item()

            current_values = self.critic(all_states)
            critic_loss = F.smooth_l1_loss(all_targets, current_values)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()
            avg_critic_loss += critic_loss.item()

        self.actor_loss_list.append(avg_actor_loss / ppo_epochs)
        self.critic_loss_list.append(avg_critic_loss / ppo_epochs)

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs, reliability_requirement_list):
        num_sfc = len(sfc_list)
        env.clear()
        for i in range(num_sfc):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            reliability_requirement = reliability_requirement_list[i]
            state = (net_state.to(self.device),
                     sfc_state.to(self.device),
                     source_dest_node_pair.to(self.device),
                     reliability_requirement.to(self.device))

            with torch.no_grad():
                logits, probs = self.actor([state])

            action = self.select_action(logits, exploration=False)  # action_dim: batch_size * max_sfc_length * 1
            placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i] + reliability_requirement.tolist()
            env.step(sfc, placement)

class DRLSFCP(nn.Module):
    def __init__(self, cfg, env, sfc_generator, device='cpu'):
        super().__init__()
        self.node_state_dim, self.vnf_state_dim, _, _ = env.get_state_dim(sfc_generator)
        self.num_nodes = env.num_nodes
        self.episode = cfg.episode
        self.batch_size = cfg.batch_size
        self.max_sfc_length = cfg.max_sfc_length

        self.actor = DecoderActor(self.node_state_dim, self.vnf_state_dim, self.num_nodes, self.max_sfc_length, embedding_dim=64).to(device)
        self.critic = DecoderCritic(self.node_state_dim, self.vnf_state_dim, self.num_nodes, self.max_sfc_length, embedding_dim=64).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)

        self.replay_buffer = ReplayBuffer(capacity=self.episode * self.batch_size)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.avg_episode_reward = 0
        self.avg_acceptance_ratio = 0
        self.avg_node_resource_utilization = 0

    def select_action(self, logits, exploration=True):
        if exploration:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        return action

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        self.avg_acceptance_ratio = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list, source_dest_node_pairs, reliability_requirement_list = sfc_generator.get_sfc_states()
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                reliability_requirement = reliability_requirement_list[i]
                state = (net_state.to(self.device),
                         sfc_state.to(self.device),
                         source_dest_node_pair.to(self.device),
                         reliability_requirement.to(self.device))

                with torch.no_grad():
                    logits, probs = self.actor([state])

                action = self.select_action(logits, exploration=True)
                placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i] + reliability_requirement.tolist()
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_reliability_requirement = reliability_requirement_list[i + 1]
                    next_state = (next_net_state.to(self.device),
                                  next_sfc_state.to(self.device),
                                  next_source_dest_node_pair.to(self.device),
                                  next_reliability_requirement.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            self.avg_acceptance_ratio += env.sfc_placed_num
            env.clear()

            self.actor._last_hidden_state = None
            self.critic._last_hidden_state = None

        return self.avg_episode_reward / episode, self.avg_acceptance_ratio / sfc_generator.batch_size / episode

    def train(self, episode=1, discount=0.99):
        batch_size = self.episode * self.batch_size

        self.actor_loss_list.clear()
        self.critic_loss_list.clear()

        for e in range(episode):
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*self.replay_buffer.sample(batch_size))
            states = list(batch_states)  # states = [(net_state, sfc_state, source_dest_node_pair), (net_state, sfc_state, source_dest_node_pair)...]
            actions = torch.stack(batch_actions, dim=0).squeeze(1).to(self.device)  # actions = torch.tensor((action), (action)...)) batch_size * max_sfc_length
            rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            next_states = list(batch_next_states)
            dones = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(self.device)

            current_V = self.critic(states) # batch_size * max_sfc_length * 1

            with torch.no_grad():
                target_V = self.critic(next_states)
                target_V = rewards + ((1 - dones) * discount * target_V)

            critic_loss = F.mse_loss(current_V, target_V)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()
            self.critic_loss_list.append(critic_loss.item())

            logits, _ = self.actor(states)
            log_probs = F.log_softmax(logits, dim=-1)   # batch_size * max_sfc_length * num_nodes
            log_pi_action = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1) # get the log probs for the actions: batch_size * max_sfc_length
            log_pi_action = log_pi_action.sum(dim=1, keepdim=True)

            with torch.no_grad():
                baseline = self.critic(states)
                advantage = target_V - baseline  # 使用完整的 TD Advantage
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # 标准化

            actor_loss = - (log_pi_action * advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()
            self.actor_loss_list.append(actor_loss.item())

            self.actor._last_hidden_state = None
            self.critic._last_hidden_state = None

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs, reliability_requirement_list):
        num_sfc = len(sfc_list)
        env.clear()
        for i in range(num_sfc):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            reliability_requirement = reliability_requirement_list[i]
            state = (net_state.to(self.device),
                     sfc_state.to(self.device),
                     source_dest_node_pair.to(self.device),
                     reliability_requirement.to(self.device))

            with torch.no_grad():
                logits, probs = self.actor([state])

            action = self.select_action(logits, exploration=False)  # action_dim: batch_size * max_sfc_length * 1
            placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i] + reliability_requirement.tolist()
            env.step(sfc, placement)

        self.actor._last_hidden_state = None
        self.critic._last_hidden_state = None
