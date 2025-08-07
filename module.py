import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch
import config

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):  # hidden_dim = N * num_heads
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1)

    def forward(self, data):
        # data = Batch.from_data_list(net_states_list)
        # net_state_list = [Data(x, edge_list), Data(x, edge_list)...]
        x = data.x
        edge_index = data.edge_index

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
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

class TransformerEncoder(nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=4, dim_feedforward=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)

    def forward(self, sfc_state, src_key_padding_mask=None):
        return self.encoder(sfc_state, src_key_padding_mask=src_key_padding_mask)

class StateNetwork(nn.Module):
    def __init__(self,num_nodes, net_state_dim, vnf_state_dim, hidden_dim=64):
        super().__init__()
        self.net_state_dim = net_state_dim
        self.sfc_attention = TransformerEncoder(dim_model=hidden_dim)
        self.pos_embed = nn.Embedding(config.MAX_SFC_LENGTH + 2, hidden_dim)
        self.sfc_linear = nn.Linear(vnf_state_dim, hidden_dim)
        self.net_linear = nn.Linear(4 * net_state_dim, hidden_dim)
        self.node_embed = nn.Embedding(num_nodes, vnf_state_dim, dtype=torch.float32)

        self.query = nn.Linear(net_state_dim, 4 * net_state_dim)
        self.key = nn.Linear(net_state_dim, 4 * net_state_dim)
        self.value = nn.Linear(net_state_dim, 4 * net_state_dim)

    def forward(self, state):
        net_state, sfc_state, source_dest_node_pair = zip(*state)
        net_states_list = list(net_state)
        sfc_states_list = list(sfc_state)

        # net state self attention
        batch_net_state = Batch.from_data_list(net_states_list)  # net state = DataBatch(x, edge_index, batch, ptr)
        batch_net_state, _ = to_dense_batch(batch_net_state.x, batch_net_state.batch)    # batch_size * num_nodes * net_state_dim

        Q = self.query(batch_net_state)     # batch_size * num_nodes * net_state_dim
        K = self.key(batch_net_state)   # batch_size * num_nodes * net_state_dim
        V = self.value(batch_net_state) # batch_size * num_nodes * net_state_dim

        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.net_state_dim ** 0.5) # batch_size * num_nodes * num_nodes
        attn_weights = torch.softmax(scores, dim=-1)    # batch_size * num_nodes * num_nodes
        batch_net_state = torch.matmul(attn_weights, V) # batch_size * num_nodes * net_state_dim
        batch_net_state = self.net_linear(batch_net_state)  # batch_size * num_nodes * hidden_dim

        batch_sfc_state = torch.stack(sfc_states_list, dim=0)

        batch_source_dest_node_pair = torch.stack(source_dest_node_pair, dim=0).unsqueeze(2)   # batch_size * 2 * 1
        batch_node_pair_emb = self.node_embed(batch_source_dest_node_pair.squeeze(-1).to(torch.long)) # batch_size * 2 * vnf_state_dim
        batch_src_emb = batch_node_pair_emb[:, 0:1, :]
        batch_dst_emb = batch_node_pair_emb[:, 1:2, :]

        batch_sfc = torch.cat((batch_src_emb, batch_sfc_state, batch_dst_emb), dim=1)   # batch_size * (max_sfc_length + 2) * vnf_state_dim

        mask = (batch_sfc.abs().sum(dim=-1) == 0)   # batch_size * (max_sfc_length + 2)

        batch_sfc = self.sfc_linear(batch_sfc)    # batch_size * (max_sfc_length + 2) * hidden_dim

        # position embedding
        batch_size, seq_len, _ = batch_sfc.shape
        device = batch_sfc.device
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embed = self.pos_embed(pos_ids) # batch_size * (max_sfc_length + 2) * hidden_dim
        batch_sfc = batch_sfc + pos_embed

        batch_sfc_attention = self.sfc_attention(batch_sfc, src_key_padding_mask=mask) # batch_size * (max_sfc_length + 2) * hidden_dim

        # batch_size * (node_num + max_sfc_length + 2) * hidden_dim
        batch_state = torch.cat((batch_net_state, batch_sfc_attention), dim=1)
        return batch_state

class GCNConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=128, num_layers=3, batch_norm=True, dropout_prob=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        for layer_id in range(self.num_layers):
            if self.num_layers == 1:
                conv = GCNConv(input_dim, output_dim)
            elif layer_id == 0:
                conv = GCNConv(input_dim, embedding_dim)
            elif layer_id == num_layers - 1:
                conv = GCNConv(embedding_dim, output_dim)
            else:
                conv = GCNConv(embedding_dim, embedding_dim)

            norm_dim = output_dim if layer_id == num_layers - 1 else embedding_dim
            norm = nn.BatchNorm1d(norm_dim) if batch_norm else nn.Identity()
            dout = nn.Dropout(dropout_prob) if dropout_prob < 1. else nn.Identity()

            self.add_module('conv_{}'.format(layer_id), conv)
            self.add_module('norm_{}'.format(layer_id), norm)
            self.add_module('dout_{}'.format(layer_id), dout)

        self._init_parameters()

    def _init_parameters(self):
        for layer_id in range(self.num_layers):
            nn.init.orthogonal_(getattr(self, f'conv_{layer_id}').lin.weight)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer_id in range(self.num_layers):
            conv = getattr(self, 'conv_{}'.format(layer_id))
            norm = getattr(self, 'norm_{}'.format(layer_id))
            dout = getattr(self, 'dout_{}'.format(layer_id))
            x = conv(x, edge_index)
            if layer_id == self.num_layers - 1:
                x = dout(norm(x))
            else:
                x = F.leaky_relu(dout(norm(x)))
        return x

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden shape: 1 * batch_size * hidden_dim
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # batch_size * seq_len * hidden_dim
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.permute(1, 0, 2).repeat(1, seq_len, 1)   # batch_size * seq_len * hidden_dim
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2))) # batch_size * seq_len * hidden_dim
        attn_weights = F.softmax(self.v(energy).squeeze(2), dim=1)  # batch_size * seq_len
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs) # batch_size, 1, hidden_dim
        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, vnf_state_dim, embedding_dim=64):
        super().__init__()
        self.emb = nn.Linear(vnf_state_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)    # seq_len * batch_size * vnf_state_dim
        x = self.emb(x)
        embeddings = F.relu(x)
        outputs, hidden_state = self.gru(embeddings)    # all timestep output: seq_len * batch_size * embedding_dim, last hidden_state: 1, batch_size, embedding_dim
        return outputs, hidden_state
