import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from module import StateNetwork, GCNConvNet, Attention, Encoder, TransformerEncoder
import config

class StateNetworkActor(nn.Module):
    def __init__(self, num_nodes, net_state_dim, vnf_state_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.state_network = StateNetwork(net_state_dim, vnf_state_dim)
        # self.state_l1 = nn.Linear(num_nodes + config.MAX_SFC_LENGTH + 2, hidden_dim)
        # self.node_fc = nn.Linear(input_dim, output_dim)

    def forward(self, state, mask=None):
        state_attention = self.state_network(state, mask)
        net_tokens = state_attention[:, :self.num_nodes, :] # batch_size * num_nodes * vnf_state_dim
        sfc_tokens = state_attention[:, self.num_nodes:, :] # batch_size * max_sfc_length * vnf_state_dim
        logits = torch.matmul(sfc_tokens, net_tokens.transpose(1, 2))
        probs = F.softmax(logits, dim=-1)
        return logits, probs

class Seq2SeqActor(nn.Module):
    def __init__(self, vnf_state_dim, hidden_dim, num_layers, num_nodes):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim

        self.node_linear = nn.Linear(1, vnf_state_dim)  # batch_size * 2 * vnf_state_dim
        self.sfc_linear = nn.Linear(config.MAX_SFC_LENGTH + 2, config.MAX_SFC_LENGTH)
        self.embedding = nn.Linear(vnf_state_dim, hidden_dim)   # input: batch_size * (max_sfc_length + 2) * vnf_state_dim
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
        self.fc_out = nn.Linear(hidden_dim, num_nodes)  # output: batch_size * (max_sfc_length + 2) * num_nodes

    def forward(self, state):
        _, sfc_state, source_dest_node_pair = zip(*state)
        if len(sfc_state) > 1:
            sfc_state = torch.stack(sfc_state, dim=0)
            source_dest_node_pair = torch.stack(source_dest_node_pair, dim=0).unsqueeze(2)
        else:
            sfc_state = sfc_state[0].unsqueeze(0)
            source_dest_node_pair = source_dest_node_pair[0].unsqueeze(0)
        batch_size = sfc_state.shape[0]
        sfc_state = sfc_state.view(batch_size, config.MAX_SFC_LENGTH, self.vnf_state_dim)
        source_dest_node_pair = source_dest_node_pair.view(batch_size, 2, 1)
        source_dest_node_pair = self.node_linear(source_dest_node_pair)  # batch_size * 2 * vnf_state_dim
        sfc = torch.cat((sfc_state, source_dest_node_pair),
                        dim=1).transpose(1, 2)  # batch_size * vnf_state_dim * (max_sfc_length + 2)
        sfc = self.sfc_linear(sfc).transpose(1, 2)  # batch_size * max_sfc_length * vnf_state_dim
        embedded = self.embedding(sfc)  # batch_size * max_sfc_length * hidden_dim
        encoder_outputs, (h, c) = self.encoder(embedded)  # output, hidden state, cell state
        decoder_outputs, _ = self.decoder(encoder_outputs, (h, c))
        logits = self.fc_out(decoder_outputs)   # batch_size * max_sfc_length * num_nodes
        probs = F.softmax(logits, dim=-1)
        return logits, probs

class DecoderActor(nn.Module):
    def __init__(self, net_state_dim, vnf_state_dim, embedding_dim=64):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim
        self.node_linear = nn.Linear(1, vnf_state_dim)  # source_dest_node_pair, batch_size * 2 * vnf_state_dim
        self.encoder = Encoder(vnf_state_dim, embedding_dim=embedding_dim)  # sfc_state
        self.gcn = GCNConvNet(net_state_dim, embedding_dim) # net_state
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten()
        )   # net_state
        self.att = Attention(embedding_dim) # placement
        self.gru = nn.GRU(embedding_dim, embedding_dim) # decoder

    def forward(self, state):
        net_state, sfc_state, source_dest_node_pair = zip(*state)
        net_state_list = list(net_state)
        batch_net_state = Batch.from_data_list(net_state_list)

        if len(sfc_state) > 1:
            sfc_state = torch.stack(sfc_state, dim=0)
            source_dest_node_pair = torch.stack(source_dest_node_pair, dim=0).unsqueeze(2)
        else:
            sfc_state = sfc_state[0].unsqueeze(0)
            source_dest_node_pair = source_dest_node_pair[0].unsqueeze(0)

        batch_size = sfc_state.shape[0]
        sfc_state = sfc_state.view(batch_size, config.MAX_SFC_LENGTH, self.vnf_state_dim)
        source_dest_node_pair = source_dest_node_pair.view(batch_size, 2, 1)
        source_dest_node_pair = self.node_linear(source_dest_node_pair)

        net_state = self.gcn(batch_net_state)
        encoder_input = torch.cat((sfc_state, source_dest_node_pair), dim=1)    # batch_size * (max_sfc_length + 2) * vnf_state_dim
        encoder_output, encoder_hidden_state = self.encoder(encoder_input)  # (max_sfc_length + 2) * batch_size * embedding_dim, 1 * batch_size * embedding_dim

        hidden_state = encoder_hidden_state

        placement_logits_list = []

        query = hidden_state[-1].unsqueeze(1)   # batch_size * 1 * embedding_dim
        seq_len = encoder_output.size(0)

        for t in range(config.MAX_SFC_LENGTH):
            repeated_query = query.expand(-1, seq_len, -1)  # batch_size * seq_len * embedding_dim
            context, attn = self.att(repeated_query, encoder_output)    # context: batch_size * 1 * embedding_dim
            gru_input = context.permute(1, 0, 2)    # 1 * batch_size * embedding_dim

            output, hidden_state = self.gru(gru_input, hidden_state)    # 1 * batch_size * embedding_dim

            decoder_output = output.squeeze(0)  # batch_size * embedding_dim

            scores = torch.matmul(net_state, decoder_output.t())    # num_nodes * batch_size
            scores = scores.permute(1, 0)   # batch_size * num_nodes
            placement_logits_list.append(scores)

            query = output.permute(1, 0, 2) # batch_size * 1 * embedding_dim

        all_logits = torch.stack(placement_logits_list, dim=1)  # batch_size * max_sfc_length * num_nodes
        all_probs = F.softmax(all_logits, dim=-1)

        return all_logits, all_probs