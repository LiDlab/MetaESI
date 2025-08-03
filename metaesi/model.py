import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

import numpy as np
from scipy.sparse import diags


"""
MetaESI: A hybrid neural network for E3-Substrate Interaction (ESI) prediction combining Truncated Transformer and Graph Attention Networks.

Key Components:

1. Attention Mechanisms:
   - ScaledDotProductAttention: Computes attention scores with scaling factor
   - MultiHeadAttention: Implements multi-head self-attention with 8 heads
   - EncoderLayer: Combines attention and feed-forward layers with residual connections

2. Truncated Transformer Module:
   - Processes substrate sequences with local attention (receptive field=7)
   - Uses layer normalization and residual connections
   - Handles variable-length sequences

3. Graph Attention Network:
   - 3-layer GAT architecture with different head configurations (4,2,2)
   - Processes E3 structure as graph (nodes=residues, edges=contacts)
   - Uses LeakyReLU activation between layers

4. MetaESI Model:
   - Integrates GAT (for E3) and Truncated Transformer (for substrate)
   - Computes interface map via matrix multiplication
   - Outputs probabilities with sigmoid activation

Parameters:
- in_features: Input feature dimension (default 1280 for ESM-2 embeddings)
- Hidden dimensions: 256 (Transformer), 32/64/128 (GAT)
- Attention heads: 8 (Transformer), 4/2/2 (GAT layers)

Inputs:
- e3_e: E3 residue embeddings
- e3_a: E3 contact map adjacency matrix  
- sub_e: Substrate region embeddings

Output:
- i_map: Interface map between E3 and substrate residues
"""

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(32)
        scores.masked_fill_(attn_mask.to(scores.device), -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # context: [n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(in_features, 32 * 8, bias=False)
        self.W_K = nn.Linear(in_features, 32 * 8, bias=False)
        self.W_V = nn.Linear(in_features, 32 * 8, bias=False)
        self.fc = nn.Linear(32 * 8, 256, bias=False)
        self.norm = nn.LayerNorm(256)

    def forward(self, input_Q, input_K, input_V, attn_mask):

        Q = self.W_Q(input_Q).view(-1, 8, 32).transpose(0, 1)
        K = self.W_K(input_K).view(-1, 8, 32).transpose(0, 1)
        V = self.W_V(input_V).view(-1, 8, 32).transpose(0, 1)

        # [n_heads, len_seq, d_k]
        attn_mask = attn_mask.unsqueeze(0).repeat(8, 1, 1)

        # context: [n_heads, len_q, d_v]
        # attn: [n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(0, 1).reshape(-1, 8 * 32)
        output = self.fc(context)
        # return nn.LayerNorm(d_model).cuda()(output + residual), attn
        return self.norm(output), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, in_features, bias=False)
        )
        self.norm = nn.LayerNorm(in_features)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        # return nn.LayerNorm(d_model).cuda()(output+residual)
        return self.norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, in_features):
        super(EncoderLayer, self).__init__()
        # self.emb = nn.Linear(in_features, 400)
        # self.enc_self_attn = MultiHeadAttention(400)
        # self.pos_ffn = PoswiseFeedForwardNet(256, 512)

        self.enc_self_attn = MultiHeadAttention(in_features)
        self.pos_ffn = PoswiseFeedForwardNet(256, 512)

    def forward(self, sub_inputs, attn_mask):
        #enc_inputs: [batch_size, src_len, d_model]
        #enc_self_attn_mask: [batch_size, src_len, src_len]

        #enc_outputs: [batch_size, src_len, d_model]
        #attn: [batch_size, n_heads, src_len, src_len]
        # sub_inputs = self.emb(sub_inputs)
        sub_outputs, attn = self.enc_self_attn(sub_inputs, sub_inputs, sub_inputs, attn_mask)
        sub_outputs = self.pos_ffn(sub_outputs)
        return sub_outputs, attn


class Transformer(nn.Module):
    def __init__(self, in_features):
        super(Transformer, self).__init__()
        self.stream0 = nn.ModuleList([EncoderLayer(in_features) for _ in range(1)])

    def forward(self, sub):
        # Create a banded diagonal attention mask to restrict each position's attention
        # to only nearby positions within a limited range.
        #
        # If sequence length exceeds the receptive field length (receptive_field_len=7),
        # create a banded mask that only allows attention within the local window.
        # Otherwise, create a matrix of all ones (allowing full attention to all positions).

        sub_len = sub.size()[0]
        receptive_field_len = 7
        if sub_len > receptive_field_len:
            attn_mask =  torch.tensor(np.array(diags([1]*receptive_field_len, list(range(int(-receptive_field_len/2), int(receptive_field_len/2) + 1)), shape=(sub_len, sub_len)).toarray()))
        else:
            attn_mask  = torch.tensor(np.ones([sub_len, sub_len]))
        attn_mask = attn_mask.data.eq(0)

        for layer in self.stream0:
            sub, enc_self_attn0 = layer(sub, attn_mask)

        return sub


class MetaESI(nn.Module):
    def __init__(self, in_features = 1280):
        super(MetaESI, self).__init__()

        self.trans = Transformer(in_features)

        self.gat1 = GATConv(in_features, 32, 4)
        self.gat2 = GATConv(128, 32, 2)
        self.gat3 = GATConv(64, 32, 2)

        self.act = nn.LeakyReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, e3_e, e3_a, sub_e):

        e3_e_1 = self.act(self.gat1(e3_e, e3_a))
        e3_e_2 = self.act(self.gat2(e3_e_1, e3_a))
        e3_e_3 = self.act(self.gat3(e3_e_2, e3_a))
        e3 = torch.cat([e3_e_1, e3_e_2, e3_e_3], dim=1)

        sub = self.act(self.trans(sub_e))
        sub = torch.transpose(sub, dim0=1, dim1=0)

        i_map = torch.mm(e3, sub)
        i_map = self.act2(i_map)

        return i_map
