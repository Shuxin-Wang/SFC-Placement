import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import networkx as nx
from torch.nn.functional import batch_norm
from torch_geometric.data import Data
import config
import environment
from model import StateNetwork
from sfc import SFCBatchGenerator

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

# todo: mask delivery
class StateNetworkActor(nn.Module):
    def __init__(self, net_state_dim, vnf_state_dim, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.state_network = StateNetwork(net_state_dim, vnf_state_dim)
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.layer_norm = nn.LayerNorm(action_dim)

    def forward(self, state, mask=None):
        x = self.state_network(state, mask)
        x = torch.flatten(x, start_dim=1)
        x = self.l1(x)
        x = self.l2(F.relu(x))
        x = self.l3(F.relu(x))
        action = self.layer_norm(x)
        return action

    # convert actor action output to sfc placement
    @staticmethod
    def get_sfc_placement(actions, node_num):
        # batch_size * 1 * max_sfc_length / 1 * max_sfc_length
        min_values, _ = actions.view(actions.size(0), -1).min(dim=1)
        max_values, _ = actions.view(actions.size(0), -1).max(dim=1)

        # batch_size * 1
        min_values = min_values.unsqueeze(1)
        max_values = max_values.unsqueeze(1)

        # batch_size * max_sfc_length
        placement = (node_num - 1) * (actions - min_values) / (max_values - min_values + 1e-8)
        placement = placement.round().int()

        return placement

class StateNetworkCritic(nn.Module):
    def __init__(self, net_state_dim, vnf_state_dim, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.state_network = StateNetwork(net_state_dim, vnf_state_dim)
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, mask=None):
        state = self.state_network(state, mask)
        state = torch.flatten(state, start_dim=1)
        x = torch.cat((state, action), dim=1)
        x = self.l1(x)
        x = self.l2(F.relu(x))
        q = self.l3(F.relu(x))
        return q

class DDPG:
    def __init__(self, node_state_dim, vnf_state_dim, state_output_dim, action_dim, device='cpu'):
        self.actor = StateNetworkActor(node_state_dim, vnf_state_dim, state_output_dim, action_dim).to(device)
        self.target_actor = StateNetworkActor(node_state_dim, vnf_state_dim, state_output_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = StateNetworkCritic(node_state_dim, vnf_state_dim, state_output_dim, action_dim).to(device)
        self.target_critic = StateNetworkCritic(node_state_dim, vnf_state_dim, state_output_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.episode_reward_list = []

    def select_action(self, state, noise_scale=0.1, exploration=True):
        action = self.actor(state)

        if exploration:
            noise = torch.randn_like(action) * noise_scale
            action = action + noise

        return action

    def fill_replay_buffer(self, env, sfc_generator, episode):
        self.episode_reward_list.clear()
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            episode_reward = 0
            for i in range(sfc_generator.batch_size):  # each episode contains batch_size sfc
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                state = (net_state.to(self.device), sfc_state.to(self.device))

                with torch.no_grad():
                    action = self.select_action([state]).squeeze(0)  # action_dim: max_sfc_length

                placement = self.actor.get_sfc_placement(action.unsqueeze(0),
                                                          env.num_nodes)  # action_dim: 1 * max_sfc_length
                sfc = sfc_list[i]
                placement = placement[0][:len(sfc_list[i])].squeeze(0)  # masked placement
                next_node_states, reward = env.step(sfc, placement)

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_state = (next_net_state.to(self.device), next_sfc_state.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward
                env.clear_sfc()
            self.episode_reward_list.append(episode_reward)
            env.clear()


    def train(self, episode, batch_size=10, discount=0.99, tau=0.005):

        self.actor_loss_list.clear()
        self.critic_loss_list.clear()

        for e in range(episode):
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*self.replay_buffer.sample(batch_size))
            states = list(batch_states)  # states = [(net_state, sfc_state), (net_state, sfc_state)...]
            actions = torch.stack(batch_actions, dim=0).to(self.device)  # actions = torch.tensor((action), (action)...)) batch_size * max_sfc_length
            rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            next_states = list(batch_next_states)
            dones = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(self.device)

            actor_action = self.actor(states)
            actor_loss = -self.critic(states, actor_action.detach()).mean()
            self.actor_loss_list.append(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()

            current_Q = self.critic(states, actions)

            with torch.no_grad():
                next_action = self.target_actor(next_states)
                target_Q = self.target_critic(next_states, next_action)
                target_Q = rewards + ((1 - dones) * discount * target_Q)

            loss_function = nn.MSELoss()
            critic_loss = loss_function(current_Q, target_Q)
            self.critic_loss_list.append(critic_loss.item())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
            self.critic_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# todo: fit the environment
class Seq2SeqActor(nn.Module):
    def __init__(self, vnf_state_dim, hidden_dim, num_layers, num_nodes):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim

        self.embedding = nn.Linear(vnf_state_dim, hidden_dim)   # input: batch_size * max_sfc_length * vnf_state_dim
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, num_nodes)  # output: batch_size * max_sfc_length * num_nodes

    def forward(self, state):
        _, sfc_state = zip(*state)
        if len(sfc_state) > 1:
            sfc_state = torch.stack(sfc_state, dim=0)
        else:
            sfc_state = sfc_state[0].unsqueeze(0)
        batch_size = sfc_state.shape[0]
        sfc_state = sfc_state.view(batch_size, config.MAX_SFC_LENGTH, self.vnf_state_dim)
        embedded = self.embedding(sfc_state)  # input: batch_size * max_sfc_length * vnf_state_dim
        encoder_outputs, (h, c) = self.encoder(embedded)    # output, hidden state, cell state
        decoder_outputs, _ = self.decoder(encoder_outputs, (h, c))
        logits = self.fc_out(decoder_outputs)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

class ValueBaseline(nn.Module):
    def __init__(self, vnf_state_dim, hidden_dim):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim
        self.embedding = nn.Linear(vnf_state_dim, hidden_dim)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        _, sfc_state = zip(*state)
        if len(sfc_state) > 1:
            sfc_state = torch.stack(sfc_state, dim=0)
        else:
            sfc_state = torch.tensor(sfc_state[0]).unsqueeze(0)
        embedded = self.embedding(sfc_state)
        _, (h, _) = self.encoder(embedded)
        value = self.fc_out(h[-1])
        return value    # batch_size * 1

class NCO(nn.Module):
    def __init__(self, vnf_state_dim, num_nodes, device='cpu'):
        super().__init__()

        self.actor = Seq2SeqActor(vnf_state_dim, hidden_dim=8, num_layers=2, num_nodes=num_nodes).to(device)
        self.critic = ValueBaseline(vnf_state_dim, hidden_dim=8).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.device = device

        self.episode_reward_list = []

    def select_action(self, state, noise_scale=0.1, exploration=True):
        _, probs = self.actor(state)
        if exploration:
            noise = torch.randn_like(probs) * noise_scale
            probs_noised = probs + noise
            action = torch.argmax(probs_noised, dim=-1)
        else:
            action = torch.argmax(probs, dim=-1)
        return action

    def fill_replay_buffer(self, env, sfc_generator, episode):
        self.episode_reward_list.clear()
        for e in range(episode):
            sfc_list = sfc_generator.get_sfc_batch()
            sfc_state_list = sfc_generator.get_sfc_states()
            episode_reward = 0
            for i in range(sfc_generator.batch_size):
                aggregate_features = env.aggregate_features()  # get aggregated node features
                edge_index = env.get_edge_index()
                net_state = Data(x=aggregate_features, edge_index=edge_index)
                sfc_state = sfc_state_list[i]
                state = (net_state.to(self.device), sfc_state.to(self.device))

                with torch.no_grad():
                    action = self.select_action([state])

                placement = action[:len(sfc_list[i])].squeeze(0)   # masked placement
                sfc = sfc_list[i]
                # print(placement, sfc)
                next_node_states, reward = env.step(sfc, placement)

                if i + 1 >= sfc_generator.batch_size:
                    next_state = state
                    done = torch.tensor(1, dtype=torch.float32)
                else:
                    next_net_state = Data(x=next_node_states, edge_index=edge_index)
                    next_sfc_state = sfc_state_list[i + 1]
                    next_state = (next_net_state.to(self.device), next_sfc_state.to(self.device))
                    done = torch.tensor(0, dtype=torch.float32)

                self.replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward
                env.clear_sfc()
            self.episode_reward_list.append(episode_reward)
            env.clear()

    def train(self, episode, batch_size=10, discount=0.99, tau=0.005):

        self.actor_loss_list.clear()
        self.critic_loss_list.clear()

        for e in range(episode):
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(
                *self.replay_buffer.sample(batch_size))
            states = list(batch_states)  # state = [(net_state, sfc_state), (net_state, sfc_state)...]
            actions = torch.stack(batch_actions, dim=0).to(
                self.device)  # action = torch.tensor((action), (action)...)) batch_size * max_sfc_length
            rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            next_states = list(batch_next_states)
            dones = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(self.device)

            _, probs = self.actor(states)
            log_probs = torch.log(probs).mean(dim=1, keepdim=True)

            with torch.no_grad():
                baseline = self.critic(states)
            advantage = rewards - baseline
            print(rewards.shape)
            print(baseline.shape)
            print(advantage.shape)
            print(log_probs.shape)
            actor_loss = -(advantage * log_probs).mean()

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

if __name__ == '__main__':
    # todo: optimize code
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    G = nx.read_graphml('Cogentco.graphml')
    env = environment.Environment(G)
    sfc_generator = SFCBatchGenerator(20, config.MIN_SFC_LENGTH, config.MAX_SFC_LENGTH,
                                          config.NUM_VNF_TYPES)

    env.clear()
    env.get_state_dim(sfc_generator)

    node_state_dim = env.node_state_dim
    vnf_state_dim = env.vnf_state_dim
    state_dim = env.state_dim
    state_input_dim = node_state_dim * env.num_nodes + config.MAX_SFC_LENGTH * vnf_state_dim
    state_output_dim = (env.num_nodes + config.MAX_SFC_LENGTH) * vnf_state_dim

    # agent = DDPG(node_state_dim, vnf_state_dim, state_output_dim,
    #              config.MAX_SFC_LENGTH, device)
    agent = NCO(vnf_state_dim, env.num_nodes, device)


    for iteration in range(config.ITERATION):
        env.clear()
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


