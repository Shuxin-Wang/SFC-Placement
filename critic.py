import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch
import torch.nn.functional as F
from module import StateNetwork, GCNConvNet, Attention, Encoder, ACEDStateNetwork
import utils

class LSTMCritic(nn.Module):
    def __init__(self, node_state_dim, vnf_state_dim, num_nodes, hidden_dim):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim
        self.node_embed = nn.Embedding(num_nodes, vnf_state_dim)
        self.reliability_fc = nn.Linear(1, vnf_state_dim)
        self.net_fc = nn.Linear(node_state_dim, hidden_dim)
        self.embedding = nn.Linear(vnf_state_dim, hidden_dim)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        net_state, sfc_state, source_dest_node_pair, reliability_requirement = utils.unpack_state(state)

        net_states_list = list(net_state)
        batch_net_state = Batch.from_data_list(net_states_list)  # net state = DataBatch(x, edge_index, batch, ptr)
        batch_net_state, _ = to_dense_batch(batch_net_state.x,
                                            batch_net_state.batch)  # batch_size * num_nodes * node_state_dim
        batch_net_state = self.net_fc(batch_net_state)  # batch_size * num_nodes * hidden_dim

        source_dest_node_pair = self.node_embed(source_dest_node_pair.squeeze(-1).to(torch.long))  # batch_size * 2 * vnf_state_dim
        reliability_requirement = self.reliability_fc(reliability_requirement)  # batch_size * 1 * vnf_state_dim
        sfc = torch.cat((sfc_state, source_dest_node_pair, reliability_requirement), dim=1)  # batch_size * (max_sfc_length + 2 + 1) * vnf_state_dim
        embedded = self.embedding(sfc)

        encoder_input = torch.cat((embedded, batch_net_state),
                                  dim=1)  # batch_size * (max_sfc_length + 2 + 1 + num_nodes) * hidden_dim

        _, (h, _) = self.encoder(encoder_input)
        value = self.fc_out(h[-1])  # h[-1]: batch_size * 1
        return value  # batch_size * 1

class DecoderCritic(nn.Module):
    def __init__(self, node_state_dim, vnf_state_dim, num_nodes, max_sfc_length, embedding_dim=64):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim
        self.max_sfc_length = max_sfc_length

        self.node_embed = nn.Embedding(num_nodes, vnf_state_dim)
        self.reliability_fc = nn.Linear(1, vnf_state_dim)
        self.encoder = Encoder(vnf_state_dim, embedding_dim=embedding_dim)  # sfc_state

        self.att = Attention(embedding_dim) # placement
        self.gcn = GCNConvNet(node_state_dim, embedding_dim) # net_state
        self.mlp = nn.Linear(embedding_dim, max_sfc_length)
        self.gru = nn.GRU(embedding_dim, embedding_dim) # decoder
        self.fc = nn.Linear(num_nodes, 1)

        self._last_hidden_state = None

    def forward(self, state):
        net_state, sfc_state, source_dest_node_pair, reliability_requirement = utils.unpack_state(state)

        net_state_list = list(net_state)
        batch_net_state = Batch.from_data_list(net_state_list)

        net_state = self.gcn(batch_net_state)
        net_state = net_state.reshape(len(state), -1, net_state.shape[-1])  # batch_size * num_nodes * embedding_dim

        sfc_state, source_dest_node_pair, reliability_requirement = utils.reshape_sfc_state_batch(sfc_state, source_dest_node_pair, reliability_requirement, self.vnf_state_dim, self.max_sfc_length)

        source_dest_node_pair = self.node_embed(source_dest_node_pair.to(torch.long))  # batch_size * 2 * vnf_state_dim
        reliability_requirement = self.reliability_fc(reliability_requirement)  # batch_size * 1 * vnf_state_dim
        sfc = torch.cat((sfc_state, source_dest_node_pair, reliability_requirement), dim=1)  # batch_size * (max_sfc_length + 2 + 1) * vnf_state_dim

        # (max_sfc_length + 2 + 1) * batch_size * embedding_dim, 1 * batch_size * embedding_dim
        encoder_output, encoder_hidden_state = self.encoder(sfc)

        if self._last_hidden_state is not None:
            net_state = net_state + self._last_hidden_state.permute(1, 0, 2)
            context, attention = self.att(self._last_hidden_state, encoder_output)
            gru_output, hidden_state = self.gru(encoder_output, self._last_hidden_state)
        else:
            net_state = net_state + encoder_hidden_state.permute(1, 0, 2)
            context, attention = self.att(encoder_hidden_state, encoder_output)
            gru_output, hidden_state = self.gru(encoder_output, encoder_hidden_state)

        self._last_hidden_state = hidden_state

        logits = self.mlp(net_state).permute(0, 2, 1)  # batch_size * max_sfc_length * num_nodes
        value = torch.mean(logits.sum(dim=-1), dim=-1, keepdim=True)
        return value

class StateNetworkCritic(nn.Module):
    def __init__(self, node_state_dim, vnf_state_dim, num_nodes, max_sfc_length, hidden_dim=256):
        super().__init__()
        self.num_nodes = num_nodes
        self.vnf_state_dim = vnf_state_dim
        self.max_sfc_length = max_sfc_length
        self.state_network = StateNetwork(node_state_dim, vnf_state_dim, num_nodes, max_sfc_length)
        self.attn = nn.Linear(num_nodes, 1)
        self.fc = nn.Sequential(
            nn.Linear(num_nodes * max_sfc_length, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        state_attention = self.state_network(state)
        net_tokens = state_attention[:, :self.num_nodes, :]  # batch_size * num_nodes * vnf_state_dim
        sfc_tokens = state_attention[:, self.num_nodes:, :]  # batch_size * (max_sfc_length + 2 + 1) * vnf_state_dim
        vnf_tokens = sfc_tokens[:, 1:-2, :]  # batch_size * max_sfc_length * hidden_dim

        logits = torch.matmul(vnf_tokens, net_tokens.transpose(1, 2))  # batch_size * max_sfc_length * num_nodes

        logits = logits.view(logits.shape[0], -1)
        q = self.fc(logits)
        return q

class ACEDCritic(nn.Module):
    def __init__(self, node_state_dim, vnf_state_dim, num_nodes, max_sfc_length, embedding_dim=64, hidden_dim=256):
        super().__init__()
        self.num_nodes = num_nodes
        self.vnf_state_dim = vnf_state_dim
        self.max_sfc_length = max_sfc_length
        self.state_network = ACEDStateNetwork(node_state_dim, vnf_state_dim, num_nodes, max_sfc_length)
        self.hidden_layer_1 = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(max_sfc_length * hidden_dim, 1)

    def forward(self, state):
        state = self.state_network(state)   # batch_size * max_sfc_length * embedding_dim
        state = self.hidden_layer_1(state)  # batch_size * max_sfc_length * hidden_dim
        state = F.leaky_relu(state)
        state = self.hidden_layer_2(state)  # batch_size * max_sfc_length * hidden_dim
        state = state.reshape(state.shape[0], -1)   # batch_size * (max_sfc_length * hidden_dim)
        state = self.fc(state)  # batch_size * 1
        value = F.tanh(state)
        return value
