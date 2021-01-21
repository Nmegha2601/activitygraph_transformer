import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.transformer_layers import GraphSelfAttention, GraphEncoderDecoderAttention


class Graph_Transformer(nn.Module):

    def __init__(self, nqueries, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=4, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        encoder_layer = GraphTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = GraphTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = GraphTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) 
        self.decoder = GraphTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)


        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask, query_embed, pos_embed):
        bs, c, t = src.shape
        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs,1)
        tgt = torch.zeros_like(query_embed)
        encoder_mask = None
        memory = self.encoder(src, mask=encoder_mask, src_key_padding_mask=src_mask, pos=pos_embed)
        ctx = memory
         
        tgt_mask = None
        hs, edge = self.decoder(tgt, ctx.permute(1,0,2), tgt_mask=tgt_mask, memory_key_padding_mask=src_mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0), edge



class GraphTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class GraphTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output, edge = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate), edge
        return output.unsqueeze(0), edge


class GraphTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False):
        super().__init__()

        self.self_attn = GraphSelfAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if src_mask:
            graph = q.permute(1,0,2) * src_mask[:,:,None]
        else:
            graph = q.permute(1,0,2)
        adj =  (torch.ones((q.size(1),q.size(0),q.size(0))))
        adj = adj.to(q.device)
        src2, _ = self.self_attn(graph, src, adj)
        src2 = src2.permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        if src_mask:
            graph = q.permute(1,0,2) * src_mask[:,:,None]
        else:
            graph = q.permute(1,0,2)

        adj =  (torch.ones((q.size(1),q.size(0),q.size(0))))
        adj = adj.to(q.device)
        src2, _ = self.self_attn(graph, pos, adj).permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class GraphTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False): 
        super().__init__()

        self.self_attn = GraphSelfAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = GraphEncoderDecoderAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)

        adj = torch.ones((q.size(1),q.size(0),q.size(0)))
        adj = adj.to(q.device)
        tgt2, edge = self.self_attn(q.permute(1,0,2), tgt, adj)
        tgt2 = tgt2.permute(1,0,2)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        st_adj = torch.ones((q.size(1),q.size(0),memory.size(1))).to(q.device)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos).permute(1,0,2), self.with_pos_embed(memory,pos.permute(1,0,2)), memory, tgt, st_adj).permute(1,0,2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, edge

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)

        adj = torch.ones((q.size(0),q.size(0)))
        adj = adj.to(q.device)
        tgt2, edge = self.self_attn(q.permute(1,0,2), adj).permute(1,0,2)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        st_adj = torch.ones((q.size(0),memory.size(1))).to(q.device)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos).permute(1,0,2), self.with_pos_embed(memory,pos.permute(1,0,2)), memory, tgt, st_adj).permute(1,0,2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, edge

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
     return Graph_Transformer(
       nqueries=args.num_queries,
       d_model=args.hidden_dim,
       dropout=args.dropout,
       nhead=args.nheads,
       dim_feedforward=args.dim_feedforward,
       num_encoder_layers=args.enc_layers,
       num_decoder_layers=args.dec_layers,
       activation=args.activation,
       normalize_before=args.pre_norm,
       return_intermediate_dec=True)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "elu":
        return F.elu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/elu, not {activation}.")
