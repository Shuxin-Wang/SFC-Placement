import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from environment import Environment
from sfc import SFCBatchGenerator
import config

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):  # hidden_dim = N * num_heads
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, data):
        # data = Batch.from_data_list(net_states_list)
        # net_state_list = [Data(x, edge_list), Data(x, edge_list)...]
        x = data.x
        edge_index = data.edge_index

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = self.norm(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # batch_size, num_heads, len_q, dim_k
        attention = torch.matmul(q / self.temperature, k.transpose(2, 3))  # len_k <-> dim_k
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = self.dropout(F.softmax(attention, dim=-1))
        # batch_size, num_heads, len_q, dim_v
        output = torch.matmul(attention, v)

        return output, attention


class MultiheadAttention(nn.Module):
    def __init__(self, dim_model, dim_k, dim_v, num_heads, dropout=0.1):  # dim_model = dim_k * num_heads
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = dim_model
        self.dim_q = dim_k
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.W_qs = nn.Linear(dim_model, num_heads * dim_k, bias=False)
        self.W_ks = nn.Linear(dim_model, num_heads * dim_k, bias=False)
        self.W_vs = nn.Linear(num_heads * dim_v, dim_model, bias=False)
        self.fc = nn.Linear(num_heads * dim_v, dim_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=dim_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # batch_size, len_q, dim_model
        dim_k, dim_v, num_heads = self.dim_k, self.dim_v, self.num_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # batch_size, len_q, num_heads, dim_k
        q = self.W_qs(q).view(batch_size, len_q, num_heads, dim_k)
        k = self.W_ks(k).view(batch_size, len_k, num_heads, dim_k)
        v = self.W_vs(v).view(batch_size, len_v, num_heads, dim_v)

        # batch_size, num_heads, len_q, dim_k
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # batch_size, 1, len_q, dim_k
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attention = self.attention(q, k, v, mask=mask)

        # batch_size, len_q, num_heads * dim_k
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attention

class Encoder(nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=3, dim_feedforward=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)

    def forward(self, sfc_state):
        return self.encoder(sfc_state)

class StateNetwork(nn.Module):
    def __init__(self, net_state_dim, vnf_state_dim):
        super().__init__()
        self.net_attention = GAT(input_dim=net_state_dim, hidden_dim=64, output_dim=3, num_heads=8)
        # self.sfc_attention = MultiheadAttention(dim_model=vnf_state_dim, dim_k=3, dim_v=3, num_heads=1)
        self.sfc_attention = Encoder(dim_model=vnf_state_dim)

    def forward(self, state, mask=None):
        net_state, sfc_state = zip(*state)
        net_states_list = list(net_state)
        sfc_states_list = list(sfc_state)

        batch_size = len(net_states_list)
        num_nodes = len(net_states_list[0].x)

        batch_net_state = Batch.from_data_list(net_states_list)  # net state = DataBatch(x, edge_index, batch, ptr)
        batch_sfc_state = torch.stack(sfc_states_list, dim=0)

        batch_net_attention = self.net_attention(batch_net_state)
        batch_net_attention = batch_net_attention.view(num_nodes, batch_size, -1).transpose(0, 1)   # batch_size, num_nodes, vnf_state_dim
        # batch_sfc_attention, _ = self.sfc_attention(batch_sfc_state, batch_sfc_state, batch_sfc_state, mask=mask)
        batch_sfc_attention = self.sfc_attention(batch_sfc_state)

        # batch_size * (node_num + max_sfc_length) * vnf_state_dim
        batch_state = torch.cat((batch_net_attention, batch_sfc_attention), dim=1)

        return batch_state

if __name__ == '__main__':

    random.seed(27)

    G = nx.read_graphml('Cogentco.graphml')
    env = Environment(G)
    sfc_generator = SFCBatchGenerator(config.BATCH_SIZE, config.MIN_SFC_LENGTH, config.MAX_SFC_LENGTH,
                                      config.NUM_VNF_TYPES)
    sfc_generator.get_sfc_batch()
    sfc_states = sfc_generator.get_sfc_states()
    env.get_state_dim(sfc_generator)

    node_state_dim = env.node_state_dim
    vnf_state_dim = env.vnf_state_dim

    # state network test
    aggregate_features = env.aggregate_features()
    edge_index = env.get_edge_index()
    net_state = Data(x=aggregate_features, edge_index=edge_index)
    state_network = StateNetwork(node_state_dim, vnf_state_dim)
    batch_state = [(net_state, sfc_states[0]), (net_state, sfc_states[1]), (net_state, sfc_states[2])]    # net_state = Data(x=[197, 28], edge_index=[2, 486])
    state = state_network(batch_state)
    print(state)

    # GAT test
    # aggregate_features = env.aggregate_features()
    # edge_index = env.get_edge_index()
    # net_state = Data(x=aggregate_features, edge_index=edge_index)
    # net_state_list = [net_state, net_state, net_state]
    # batch_net_state = Batch.from_data_list(net_state_list)
    # gat = GAT(input_dim=node_state_dim, hidden_dim=64, output_dim=vnf_state_dim, num_heads=8)
    # print('GAT output:', gat(batch_net_state))

    # sfc attention test
    # batch_sfc_state = torch.stack([sfc_states[0], sfc_states[0], sfc_states[0]], dim=0)
    # multihead_attention = MultiheadAttention(dim_model=vnf_state_dim, dim_k=3, dim_v=3, num_heads=1)
    # y, attention = multihead_attention(batch_sfc_state, batch_sfc_state, batch_sfc_state)
    # print('multihead attention output:', y)
    # print('attention weights:', attention)
