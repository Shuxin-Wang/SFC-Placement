import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import networkx as nx
from torch_geometric.data import Data
import config
import environment
from sfc import SFCBatchGenerator
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
    def __init__(self, vnf_state_dim, num_nodes, device='cpu'):
        super().__init__()

        self.actor = Seq2SeqActor(vnf_state_dim, hidden_dim=8, num_layers=2, num_nodes=num_nodes).to(device)
        self.critic = LSTMCritic(num_nodes, vnf_state_dim, hidden_dim=8).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(capacity=200)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.avg_episode_reward = 0

    def select_action(self, probs, exploration=True):
        if exploration:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(probs, dim=-1)
        return action   # 1 * max_sfc_length

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            source_dest_node_pairs = sfc_generator.get_source_dest_node_pairs()
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

                with torch.no_grad():
                    _, probs = self.actor([state])

                action = self.select_action(probs, exploration=True)
                placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_state = (next_net_state.to(self.device), next_sfc_state.to(self.device),
                                  next_source_dest_node_pair.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            env.clear()
        return self.avg_episode_reward / episode

    def train(self, episode, batch_size=10, discount=0.99, tau=0.005):

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

            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # rewards normalization

            logits, _ = self.actor(states)   # batch_size * max_sfc_length * node_num
            log_probs = F.log_softmax(logits, dim=-1)
            log_pi_action = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # get the log probs for the actions: batch_size * max_sfc_length
            log_pi_action = log_pi_action.sum(dim=1, keepdim=True)

            with torch.no_grad():
                baseline = self.critic(states)

            advantage = rewards - baseline
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            actor_loss = -(advantage * log_pi_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()
            self.actor_loss_list.append(actor_loss.item())

            values = self.critic(states)
            critic_loss = F.mse_loss(values, rewards)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()
            self.critic_loss_list.append(critic_loss.item())

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs):
        self.avg_episode_reward = 0
        env.clear()
        for i in range(len(sfc_list)):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

            with torch.no_grad():
                _, probs = self.actor([state])

            action = self.select_action(probs, exploration=False)  # action_dim: batch_size * max_sfc_length * 1
            placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
            _, reward = env.step(sfc, placement)
            self.avg_episode_reward += reward

class EnhancedNCO(nn.Module):
    def __init__(self, num_nodes, node_state_dim, vnf_state_dim, device='cpu'):
        super().__init__()

        self.actor = StateNetworkActor(num_nodes, node_state_dim, vnf_state_dim).to(device)
        self.critic = StateNetworkCritic(node_state_dim, vnf_state_dim, num_nodes, hidden_dim=64).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-5)

        self.replay_buffer = ReplayBuffer(capacity=200)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.episode_reward = 0

    def select_action(self, probs, exploration=True):
        if exploration:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(probs, dim=-1)
        return action

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            source_dest_node_pairs = sfc_generator.get_source_dest_node_pairs()
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

                with torch.no_grad():
                    _, probs = self.actor([state])

                action = self.select_action(probs, exploration=True)
                placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist() # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_state = (next_net_state.to(self.device), next_sfc_state.to(self.device),
                                  next_source_dest_node_pair.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            env.clear()
        return self.avg_episode_reward / episode

    def train(self, episode, batch_size=10, discount=0.99, tau=0.005):

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

            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # rewards normalization

            with torch.no_grad():
                next_values = self.critic(next_states)
                target_values = rewards + discount * next_values * (1 - dones)
                # target_values = (target_values - target_values.mean()) / (target_values.std() + 1e-8)

            current_values = self.critic(states)
            critic_loss = F.mse_loss(target_values, current_values)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()
            self.critic_loss_list.append(critic_loss.item())

            logits, _ = self.actor(states)  # batch_size * max_sfc_length * node_num
            log_probs = F.log_softmax(logits, dim=-1)
            log_pi_action = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(
                -1)  # get the log probs for the actions: batch_size * max_sfc_length
            log_pi_action = log_pi_action.sum(dim=1, keepdim=True)

            with torch.no_grad():
                baseline = self.critic(states)
                advantage = target_values - baseline  # 使用完整的 TD Advantage
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # 标准化

            actor_loss = -(advantage * log_pi_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()
            self.actor_loss_list.append(actor_loss.item())

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs):
        self.episode_reward = 0
        env.clear()
        for i in range(len(sfc_list)):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

            with torch.no_grad():
                _, probs = self.actor([state])

            action = self.select_action(probs, exploration=False)  # action_dim: batch_size * max_sfc_length * 1
            placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
            _, reward = env.step(sfc, placement)
            self.episode_reward += reward

class ActorEnhancedNCO(nn.Module):
    def __init__(self, num_nodes, node_state_dim, vnf_state_dim, device='cpu'):
        super().__init__()

        self.actor = StateNetworkActor(num_nodes, node_state_dim, vnf_state_dim).to(device)
        self.critic = LSTMCritic(num_nodes, vnf_state_dim, hidden_dim=8).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(capacity=200)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.episode_reward = 0

    def select_action(self, probs, exploration=True):
        if exploration:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(probs, dim=-1)
        return action

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            source_dest_node_pairs = sfc_generator.get_source_dest_node_pairs()
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

                with torch.no_grad():
                    _, probs = self.actor([state])

                action = self.select_action(probs, exploration=True)
                placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist() # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_state = (next_net_state.to(self.device), next_sfc_state.to(self.device),
                                  next_source_dest_node_pair.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            env.clear()
        return self.avg_episode_reward / episode

    def train(self, episode, batch_size=10, discount=0.99, tau=0.005):

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

            logits, _ = self.actor(states)   # batch_size * max_sfc_length * node_num
            log_probs = F.log_softmax(logits, dim=-1)
            log_pi_action = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # get the log probs for the actions: batch_size * max_sfc_length
            log_pi_action = log_pi_action.sum(dim=1, keepdim=True)

            with torch.no_grad():
                baseline = self.critic(states)

            advantage = rewards - baseline
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # rewards normalization

            actor_loss = -(advantage * log_pi_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()
            self.actor_loss_list.append(actor_loss.item())

            values = self.critic(states)
            critic_loss = F.mse_loss(values, rewards)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()
            self.critic_loss_list.append(critic_loss.item())

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs):
        self.episode_reward = 0
        env.clear()
        for i in range(len(sfc_list)):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

            with torch.no_grad():
                _, probs = self.actor([state])

            action = self.select_action(probs, exploration=False)  # action_dim: batch_size * max_sfc_length * 1
            placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
            _, reward = env.step(sfc, placement)
            self.episode_reward += reward

class CriticEnhancedNCO(nn.Module):
    def __init__(self, num_nodes, node_state_dim, vnf_state_dim, device='cpu'):
        super().__init__()

        self.actor = Seq2SeqActor(vnf_state_dim, hidden_dim=8, num_layers=2, num_nodes=num_nodes).to(device)
        self.critic = StateNetworkCritic(node_state_dim, vnf_state_dim, num_nodes).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(capacity=200)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.episode_reward = 0

    def select_action(self, probs, exploration=True):
        if exploration:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(probs, dim=-1)
        return action.to(dtype=torch.float32)

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            source_dest_node_pairs = sfc_generator.get_source_dest_node_pairs()
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

                with torch.no_grad():
                    _, probs = self.actor([state])

                action = self.select_action(probs, exploration=True)
                placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_state = (next_net_state.to(self.device), next_sfc_state.to(self.device),
                                  next_source_dest_node_pair.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            env.clear()
        return self.avg_episode_reward / episode

    def train(self, episode, batch_size=10, discount=0.99, tau=0.005):

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

            # _, probs = self.actor(states)   # batch_size * max_sfc_length * node_num
            # B, T, N = probs.shape
            # probs_reshaped = probs.view(B * T, N)
            #
            # dist = torch.distributions.Categorical(probs_reshaped)
            # action = dist.sample()  # every action choose a node
            # log_probs = dist.log_prob(action).view(B, T).mean(dim=1, keepdim=True)

            logits, _ = self.actor(states)   # batch_size * max_sfc_length * node_num
            log_probs = F.log_softmax(logits, dim=-1)
            log_pi_action = log_probs.gather(dim=-1, index=actions.to(dtype=torch.int64).unsqueeze(-1)).squeeze(-1)  # get the log probs for the actions: batch_size * max_sfc_length

            with torch.no_grad():
                baseline = self.critic(states, actions)

            advantage = rewards - baseline
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # rewards normalization

            actor_loss = -(advantage * log_pi_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()
            self.actor_loss_list.append(actor_loss.item())

            values = self.critic(states, actions)
            critic_loss = F.mse_loss(values, rewards)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()
            self.critic_loss_list.append(critic_loss.item())

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs):
        self.episode_reward = 0
        env.clear()
        for i in range(len(sfc_list)):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

            with torch.no_grad():
                _, probs = self.actor([state])

            action = self.select_action(probs, exploration=False)  # action_dim: batch_size * max_sfc_length * 1
            placement = torch.argmax(action, dim=-1)
            placement = placement[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
            _, reward = env.step(sfc, placement)
            self.episode_reward += reward

class DDPG:
    def __init__(self,num_nodes, node_state_dim, vnf_state_dim, device='cpu'):
        self.actor = StateNetworkActor(num_nodes, node_state_dim, vnf_state_dim).to(device)
        self.target_actor = StateNetworkActor(num_nodes, node_state_dim, vnf_state_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = StateNetworkCritic(node_state_dim, vnf_state_dim, num_nodes).to(device)
        self.target_critic = StateNetworkCritic(node_state_dim, vnf_state_dim, num_nodes).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(capacity=200)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.episode_reward = 0

    def select_action(self, probs, noise_scale=0.1, exploration=True):
        if exploration:
            noise = torch.randn_like(probs) * noise_scale
            action = probs + noise
        else:
            action = probs
        return action

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            source_dest_node_pairs = sfc_generator.get_source_dest_node_pairs()
            for i in range(sfc_generator.batch_size):  # each episode contains batch_size sfc
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

                with torch.no_grad():
                    _, probs = self.actor([state])

                action = self.select_action(probs, exploration=True)    # action_dim: batch_size * max_sfc_length * num_nodes
                placement = torch.argmax(action, dim=-1)    # batch_size * max_sfc_length
                placement = placement[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_state = (next_net_state.to(self.device), next_sfc_state.to(self.device), next_source_dest_node_pair.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            env.clear()
        return self.avg_episode_reward / episode

    def train(self, episode, batch_size=10, discount=0.99, tau=0.005):

        self.actor_loss_list.clear()
        self.critic_loss_list.clear()

        for e in range(episode):
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*self.replay_buffer.sample(batch_size))
            states = list(batch_states)  # states = [(net_state, sfc_state, source_dest_node_pair), (net_state, sfc_state, source_dest_node_pair)...]
            actions = torch.stack(batch_actions, dim=0).squeeze(1).to(self.device)  # actions = torch.tensor((action), (action)...)) batch_size * max_sfc_length * num_nodes
            rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            next_states = list(batch_next_states)
            dones = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(self.device)

            current_Q = self.critic(states, actions)

            with torch.no_grad():
                _, next_action = self.target_actor(next_states)
                target_Q = self.target_critic(next_states, next_action)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # rewards normalization
                target_Q = rewards + ((1 - dones) * discount * target_Q)

            critic_loss = F.mse_loss(current_Q, target_Q)
            for param in self.critic.parameters():
                critic_loss += 1e-4 * param.pow(2).sum()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()

            self.critic_loss_list.append(critic_loss.item())

            _, actor_actions = self.actor(states)
            actor_loss = -self.critic(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()

            self.actor_loss_list.append(actor_loss.item())

            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs):
        self.episode_reward = 0
        env.clear()
        for i in range(len(sfc_list)):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

            with torch.no_grad():
                _, probs = self.actor([state])

            action = self.select_action(probs, exploration=False)  # action_dim: batch_size * max_sfc_length * num_nodes
            placement = torch.argmax(action, dim=-1)    # batch_size * max_sfc_length
            placement = placement[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()   # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
            _, reward = env.step(sfc, placement)
            self.episode_reward += reward

class DRLSFCP:
    def __init__(self, net_state_dim, vnf_state_dim, embedding_dim=64, device='cpu'):
        super().__init__()
        self.actor = DecoderActor(net_state_dim, vnf_state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = DecoderCritic(net_state_dim, vnf_state_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(capacity=200)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.episode_reward = 0

    def select_action(self, probs, exploration=True):
        if exploration:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(probs, dim=-1)
        return action

    def fill_replay_buffer(self, env, sfc_generator, episode):
        env.clear()
        self.avg_episode_reward = 0
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            source_dest_node_pairs = sfc_generator.get_source_dest_node_pairs()
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                source_dest_node_pair = source_dest_node_pairs[i]
                state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

                with torch.no_grad():
                    _, probs = self.actor([state])

                action = self.select_action(probs, exploration=True)
                placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
                sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
                next_node_states, reward = env.step(sfc, placement)
                self.avg_episode_reward += reward

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_source_dest_node_pair = source_dest_node_pairs[i + 1]
                    next_state = (next_net_state.to(self.device), next_sfc_state.to(self.device),
                                  next_source_dest_node_pair.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                env.clear_sfc()
            env.clear()
        return self.avg_episode_reward / episode

    def train(self, episode, batch_size=10, discount=0.99):

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
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # rewards normalization
                rewards = rewards.unsqueeze(-1)
                dones = dones.unsqueeze(-1)
                target_V = rewards + ((1 - dones) * discount * target_V)

            critic_loss = F.mse_loss(current_V, target_V)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()

            self.critic_loss_list.append(critic_loss.item())

            logits, _ = self.actor(states)
            advantage = (target_V - current_V).detach()
            log_probs = F.log_softmax(logits, dim=-1)   # batch_size * max_sfc_length * num_nodes
            log_pi_action = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)   # get the log probs for the actions: batch_size * max_sfc_length
            actor_loss = - (log_pi_action * advantage.squeeze(-1)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()

            self.actor_loss_list.append(actor_loss.item())

    def test(self, env, sfc_list, sfc_state_list, source_dest_node_pairs):
        self.episode_reward = 0
        env.clear()
        for i in range(len(sfc_list)):  # each episode contains batch_size sfc
            env.clear_sfc()
            aggregate_features = env.aggregate_features()  # get aggregated node features
            edge_index = env.get_edge_index()
            net_state = Data(x=aggregate_features, edge_index=edge_index)
            sfc_state = sfc_state_list[i]
            source_dest_node_pair = source_dest_node_pairs[i]
            state = (net_state.to(self.device), sfc_state.to(self.device), source_dest_node_pair.to(self.device))

            with torch.no_grad():
                _, probs = self.actor([state])

            action = self.select_action(probs, exploration=False)  # action_dim: batch_size * max_sfc_length * 1
            placement = action[0][:len(sfc_list[i])].squeeze(0).to(dtype=torch.int32).tolist()  # masked placement
            sfc = source_dest_node_pair.to(dtype=torch.int32).tolist() + sfc_list[i]
            _, reward = env.step(sfc, placement)
            self.episode_reward += reward

if __name__ == '__main__':
    # todo: optimize code
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    G = nx.read_graphml('graph/Cogentco.graphml')
    env = environment.Environment(G)
    sfc_generator = SFCBatchGenerator(20, config.MIN_SFC_LENGTH, config.MAX_SFC_LENGTH,
                                          config.NUM_VNF_TYPES, env.num_nodes)

    env.clear()
    env.get_state_dim(sfc_generator)

    node_state_dim = env.node_state_dim
    vnf_state_dim = env.vnf_state_dim
    state_dim = env.state_dim
    state_input_dim = node_state_dim * env.num_nodes + config.MAX_SFC_LENGTH * vnf_state_dim
    state_output_dim = (env.num_nodes + config.MAX_SFC_LENGTH + 2) * vnf_state_dim

    # agent = DDPG(env.num_nodes, node_state_dim, vnf_state_dim, state_output_dim,
    #              config.MAX_SFC_LENGTH * env.num_nodes, device)
    # agent = NCO(vnf_state_dim, env.num_nodes, device)
    agent = DRLSFCP(node_state_dim, vnf_state_dim, device=device)

    for iteration in range(config.ITERATION):
        agent.fill_replay_buffer(env, sfc_generator, 10)
        agent.train(10, 20)

    # actor and critic test
    # node_features = env.aggregate_features()
    # net_state = Data(x=node_features, edge_index=edge_index)
    # sfc_generator.get_sfc_batch()
    # sfc_states = sfc_generator.get_sfc_states()
    # state = [(net_state, sfc_states[0]), (net_state, sfc_states[1]), (net_state, sfc_states[2])]
    #
    # actor = Actor(node_state_dim, vnf_state_dim, state_dim, action_dim=config.MAX_SFC_LENGTH)
    # action = actor(state)
    # print('actor action:\n', action)
    #
    # # net input dim + sfc input dim + action output dim
    # critic_input_dim = env.num_nodes * node_state_dim + config.MAX_SFC_LENGTH * vnf_state_dim + config.MAX_SFC_LENGTH
    # critic = Critic(critic_input_dim, output_dim=1)
    # q = critic(state, action)
    # print('critic q:\n', q)


