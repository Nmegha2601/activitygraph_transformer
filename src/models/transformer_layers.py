import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Norm(nn.Module):
    """ Graph Normalization """
    def __init__(self, norm_type, hidden_dim=64):
        super().__init__()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def compute_norm(self,x,dim=0):
        eps = 1e-6
        mean = x.mean(dim = dim, keepdim = True)
        var = x.std(dim = dim, keepdim = True)
        x = (x - mean) / (var + eps)
        return x


    def forward(self, x):
        if self.norm is not None and type(self.norm) != str:
            x_norm = []
            for i in range(x.size(0)):
                x_norm.append(self.compute_norm(self.compute_norm(x[i,:,:],dim=1),dim=0).unsqueeze(0))
            x = torch.cat(x_norm,dim=0)
            return x

        elif self.norm is None:
            return x
        bs, k, c = x.size() 
        batch_list = torch.tensor(1).repeat(bs).long().to(x.device)
        batch_index = torch.arange(bs).to(x.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (x.dim() - 1)).expand_as(x)
        mean = torch.zeros(bs, *x.shape[1:]).to(x.device)
        mean = mean.scatter_add_(0, batch_index, x)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = x - mean * self.mean_scale

        std = torch.zeros(bs, *x.shape[1:]).to(x.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)

        x_norm = self.weight * sub / std + self.bias
        return x_norm

class GraphEncoderDecoderAttention(nn.Module):
    def __init__(self, nhid, nheads, dropout, norm_type='bn', alpha=0.1, decoder_attn='ctx'):
        super(GraphEncoderDecoderAttention, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nheads = nheads
        self.graph_attentions = [GraphEncoderDecoderAttentionLayer(nhid, nhid, nhid//nheads, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.graph_attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.linear = nn.Linear(nhid, nhid) 
        self.norm1 = Norm(norm_type,nhid)
        self.norm2 = Norm(norm_type,nhid)
        self.activation = F.leaky_relu
        self.graph_multihead_attn = nn.MultiheadAttention(nhid, nheads, dropout=dropout)
        self.decoder_attention = decoder_attn

    def forward(self, x, ctx_with_pos, ctx,src, adj):
        x = F.dropout(x, self.dropout)
        ctx = F.dropout(ctx, self.dropout)
        x = x + torch.cat([att(x,ctx_with_pos,adj) for att in self.graph_attentions],dim=2)
        x = self.linear(self.norm1(x))
        x = F.dropout(x,self.dropout)
        x = self.norm2(x)
        x = x.permute(1,0,2)
        ctx_with_pos = ctx_with_pos.permute(1,0,2)
        ctx = ctx.permute(1,0,2)
        x = self.graph_multihead_attn(x,ctx_with_pos,value=ctx)[0]
        x = x.permute(1,0,2)
        return x

class GraphSelfAttention(nn.Module):
    def __init__(self, nhid, nheads, dropout, norm_type='bn', alpha=0.1):
        """Dense version of GAT."""
        super(GraphSelfAttention, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nheads = nheads
        self.graph_attentions = [GraphAttentionLayer(nhid, nhid//nheads, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.graph_attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.linear = nn.Linear(nhid, nhid)
        self.graph_self_attn = nn.MultiheadAttention(nhid, nheads, dropout=dropout)
        self.norm1 = Norm(norm_type,nhid)
        self.norm2 = Norm(norm_type,nhid)
        self.activation = F.leaky_relu

    def forward(self, x, src, adj):
        x = F.dropout(x, self.dropout)
        x_att = []
        e_att = []
        for att in self.graph_attentions:
            node,edge = att(x,adj)
            x_att.append(node)
            e_att.append(edge)
        x = x + torch.cat(x_att,dim=2)
        e = torch.sum(torch.stack(e_att),dim=0)/len(x_att)
        x = self.linear(self.norm1(x))
        x = F.dropout(x,self.dropout)
        x = self.norm2(x)
        x = x.permute(1,0,2)
        x = self.graph_self_attn(x,x,value=src)[0]
        x = x.permute(1,0,2)    
        return x, e



class GraphEncoderDecoderAttentionLayer(nn.Module):
    """
    Graph-to-Graph message passing, adapted from https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_src_features, in_tgt_features, out_features, dropout, alpha,  concat=True):
        super(GraphEncoderDecoderAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_src_features = in_src_features
        self.in_tgt_features = in_tgt_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.Ws = nn.Parameter(torch.empty(size=(in_src_features, out_features)))
        self.Wt = nn.Parameter(torch.empty(size=(in_tgt_features, out_features)))
        nn.init.xavier_uniform_(self.Ws.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wt.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, ctx, adj):
        Ws_ctx = torch.bmm(ctx, self.Ws.repeat(ctx.size(0),1,1)) 
        Wt_h = torch.bmm(h, self.Wt.repeat(h.size(0),1,1)) 

        a_input = self._prepare_attentional_mechanism_input(Ws_ctx, Wt_h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Ws_ctx)
        h_prime = F.leaky_relu(h_prime)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Ws_ctx, Wt_h):
        Ns = Ws_ctx.size()[1] # number of nodes
        Nt = Wt_h.size()[1] # number of nodes
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        Ws_ctx_repeated_in_chunks = Ws_ctx.repeat_interleave(Nt, dim=1)
        Wt_h_repeated_alternating = Wt_h.repeat([1,Ns,1])
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Ws_ctx_repeated_in_chunks, Wt_h_repeated_alternating], dim=2)

        return all_combinations_matrix.view(Ws_ctx.size(0),Nt, Ns, 2 * self.out_features)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha,  concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.bmm(h, self.W.repeat(h.size(0),1,1)) 
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec.to(h.device))
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        h_prime = F.leaky_relu(h_prime)
        return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1] # number of nodes
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat([1,N,1])
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        return all_combinations_matrix.view(Wh.size(0), N, N, 2 * self.out_features)


