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
        self.state_network = StateNetwork(num_nodes, net_state_dim, vnf_state_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.l_out = nn.Linear(64, num_nodes)

    def forward(self, state):
        state_attention = self.state_network(state)
        net_tokens = state_attention[:, :self.num_nodes, :] # batch_size * num_nodes * hidden_dim
        sfc_tokens = state_attention[:, self.num_nodes:, :] # batch_size * (max_sfc_length + 2) * hidden_dim
        vnf_tokens = sfc_tokens[:, 1:-1, :] # batch_size * max_sfc_length * hidden_dim

        # attn_output, _ = self.cross_attn(query=sfc_tokens, key=net_tokens, value=net_tokens)
        # logits = self.l_out(attn_output)

        logits = torch.matmul(vnf_tokens, net_tokens.transpose(1, 2))   # batch_size * max_sfc_length * num_nodes
        probs = F.softmax(logits, dim=-1)
        return logits, probs

class Seq2SeqActor(nn.Module):
    def __init__(self, vnf_state_dim, hidden_dim, num_layers, num_nodes):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim

        self.node_emb = nn.Embedding(num_nodes, vnf_state_dim)  # batch_size * 2 * vnf_state_dim
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
        source_dest_node_pair = source_dest_node_pair.view(batch_size, 2)
        source_dest_node_pair = self.node_emb(source_dest_node_pair.to(torch.long))  # batch_size * 2 * vnf_state_dim
        sfc = torch.cat((sfc_state, source_dest_node_pair),
                        dim=1)  # batch_size * (max_sfc_length + 2) * vnf_state_dim
        embedded = self.embedding(sfc)  # batch_size * (max_sfc_length + 2) * hidden_dim
        encoder_outputs, (h, c) = self.encoder(embedded)  # output, hidden state, cell state
        decoder_outputs, _ = self.decoder(encoder_outputs, (h, c))
        logits = self.fc_out(decoder_outputs[:, :config.MAX_SFC_LENGTH, :])   # batch_size * max_sfc_length * num_nodes
        probs = F.softmax(logits, dim=-1)
        return logits, probs

class DecoderActor(nn.Module):
    def __init__(self,num_nodes, net_state_dim, vnf_state_dim, embedding_dim=64):
        super().__init__()
        self.vnf_state_dim = vnf_state_dim
        self.node_embed = nn.Embedding(num_nodes, vnf_state_dim)
        self.encoder = Encoder(vnf_state_dim, embedding_dim=embedding_dim)  # sfc_state

        self.att = Attention(embedding_dim) # placement
        self.gcn = GCNConvNet(net_state_dim, embedding_dim) # net_state
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten()
        )   # net_state
        self.gru = nn.GRU(embedding_dim, embedding_dim) # decoder

    def forward(self, state):
        net_state, sfc_state, source_dest_node_pair = zip(*state)
        net_state_list = list(net_state)
        batch_net_state = Batch.from_data_list(net_state_list)

        net_state = self.gcn(batch_net_state)
        net_state = net_state.reshape(len(state), -1, net_state.shape[-1])  # batch_size * num_nodes * embedding_dim

        if len(sfc_state) > 1:
            sfc_state = torch.stack(sfc_state, dim=0)
            source_dest_node_pair = torch.stack(source_dest_node_pair, dim=0).unsqueeze(2)
        else:
            sfc_state = sfc_state[0].unsqueeze(0)
            source_dest_node_pair = source_dest_node_pair[0].unsqueeze(0)

        batch_size = sfc_state.shape[0]
        sfc_state = sfc_state.view(batch_size, config.MAX_SFC_LENGTH, self.vnf_state_dim)
        source_dest_node_pair = source_dest_node_pair.view(batch_size, 2)
        source_dest_node_pair = self.node_embed(source_dest_node_pair.to(torch.long))  # batch_size * 2 * vnf_state_dim

        encoder_input = torch.cat((sfc_state, source_dest_node_pair), dim=1)    # batch_size * (max_sfc_length + 2) * vnf_state_dim
        encoder_output, encoder_hidden_state = self.encoder(encoder_input)  # (max_sfc_length + 2) * batch_size * embedding_dim, 1 * batch_size * embedding_dim
        hidden_state = encoder_hidden_state.permute(1, 0, 2) # batch_size * 1 * embedding_dim

        placement_logits_list = []

        for t in range(config.MAX_SFC_LENGTH):
            p_node_id = torch.full((batch_size,), t, dtype=torch.long, device=sfc_state.device)  # [B]
            p_node_emb = self.node_embed(p_node_id).unsqueeze(0)

            mask = torch.zeros(batch_size, encoder_output.size(0), dtype=torch.bool, device=sfc_state.device)

            context, _ = self.att(hidden_state, encoder_output.permute(1, 0, 2), mask)    # context: batch_size * 1 * embedding_dim
            context = context.unsqueeze(0)

            output, hidden_state = self.gru(p_node_emb, hidden_state.permute(1, 0, 2))    # 1 * batch_size * embedding_dim
            hidden_state = hidden_state.permute(1, 0, 2)

            decoder_output = output.squeeze(0)  # batch_size * embedding_dim
            fused = net_state + decoder_output.unsqueeze(1) # batch_size * num_nodes * embedding_dim

            logits = self.mlp(fused)    # batch_size * num_nodes
            placement_logits_list.append(logits)

        all_logits = torch.stack(placement_logits_list, dim=1)  # batch_size * max_sfc_length * num_nodes
        all_probs = F.softmax(all_logits, dim=-1)

        return all_logits, all_probs