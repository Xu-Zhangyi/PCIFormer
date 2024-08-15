import numpy as np
import torch
import torch.nn as nn
import math
import copy
from torch_geometric.nn import GCNConv, GINConv, GATConv
from setting import SetParameter

from utils import sparse_matrix_to_tensor

args = SetParameter()


class TimeEncode(torch.nn.Module):
    r"""
     This is a trainable encoder to map continuous time value into a low-dimension time vector.
     Ref: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/master/module.py
     """

    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)
        torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)
        harmonic = harmonic.type(self.dense.weight.dtype)
        harmonic = self.dense(harmonic)
        return harmonic  # self.dense(harmonic)


class SubGraphTransformerEncoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(SubGraphTransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, q, k, v, mask=None, bias=None):
        for layer in self.layers:
            q = layer(q, k, v, mask=mask, bias=bias)
        if self.norm is not None:
            q = self.norm(q)
        return q


class SubGraphTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead,
                 dim_feedforward=256, dropout=0.3, layer_norm_eps=1e-5, norm_first=False):
        super(SubGraphTransformerEncoderLayer, self).__init__()

        self.self_attn = SubGraphMultiheadAttention(d_model, nhead, dropout)

        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first

    def forward(self, q, k, v, mask=None, bias=None):
        if self.norm_first:
            out = self.self_attn(self.norm1(q), self.norm1(k), self.norm1(v), mask=mask, bias=bias)
            q = q + self.dropout1(out)
            q = q + self.dropout2(self.feed_forward(self.norm2(q)))
        else:
            out = self.self_attn(q, k, v, mask)
            q = self.norm1_1(q + self.dropout1(out))
            q = self.norm2(q + self.dropout2(self.feed_forward(q)))
        return q


class SubGraphMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.3):
        super(SubGraphMultiheadAttention, self).__init__()

        self.embedding_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.w_bias = nn.Linear(4, 1)

        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        torch.nn.init.xavier_normal_(self.w_q.weight)
        torch.nn.init.xavier_normal_(self.w_k.weight)
        torch.nn.init.xavier_normal_(self.w_v.weight)
        torch.nn.init.xavier_normal_(self.w_o.weight)
        torch.nn.init.xavier_normal_(self.w_bias.weight)

    def forward(self, query, key, value, mask=None, bias=None):
        # q: (P,D) k,v(P,20,D)
        seq_len = query.shape[0]

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        attn = torch.sum(torch.mul(q.unsqueeze(-2), k), dim=-1) / math.sqrt(
            self.embedding_dim)  # (P,1,D)*(P,20,D)->(P,20)
        if bias is not None:
            attn = attn + self.w_bias(bias).squeeze()
        if mask is not None:
            attn = torch.softmax(attn + mask, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn_v = torch.sum(torch.mul(attn.unsqueeze(-1), v), dim=-2)  # (P,20,1)*(P,20,D)->(P,D)
        output = self.w_o(attn_v)

        return output


class SeqTransformerEncoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(SeqTransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, attn_mask=None, pad_mask=None):

        for layer in self.layers:
            x = layer(x, attn_mask, pad_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SeqTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.3, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False):
        super(SeqTransformerEncoderLayer, self).__init__()

        self.self_attn = SeqMultiheadAttention(d_model, nhead, dropout, batch_first)

        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first

    def forward(self, x, attn_mask=None, pad_mask=None):
        if self.norm_first:
            x = x + self.dropout1(
                self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask, pad_mask)[0])
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(
                self.self_attn(x, x, x, attn_mask, pad_mask)[0]))
            x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x


class SeqMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.3, batch_first=True):
        super(SeqMultiheadAttention, self).__init__()

        self.embedding_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"
        # 128 256 512 1024 2048
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        self.batch_first = batch_first

        torch.nn.init.xavier_normal_(self.w_q.weight)
        torch.nn.init.xavier_normal_(self.w_k.weight)
        torch.nn.init.xavier_normal_(self.w_v.weight)
        torch.nn.init.xavier_normal_(self.w_o.weight)

    def forward(self, query, key, value, attn_mask=None, pad_mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        q = self.w_q(query).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).reshape(
            batch_size * self.num_heads, -1, self.head_dim)
        k = self.w_k(key).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).reshape(
            batch_size * self.num_heads, -1, self.head_dim)
        v = self.w_v(value).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).reshape(
            batch_size * self.num_heads, -1, self.head_dim)

        # self.pad_mask = np.where(self.sub_poi_id == self.poi_count, '-inf', 0)
        if pad_mask is not None:
            pad_mask = pad_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1).reshape(
                batch_size * self.num_heads, 1, seq_len)
            if attn_mask is None:
                attn_mask = pad_mask
            else:
                attn_mask = attn_mask + pad_mask

        # (batch_size * num_heads, seq_len, seq_len)
        if attn_mask is not None:
            # out = input + (batch1 @ batch2)
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        else:
            attn = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attn = torch.matmul(q, k.transpose(-2, -1))
        # attn = attn / math.sqrt(self.head_dim)
        # attn += attn_mask

        attn = torch.softmax(attn, dim=-1)
        # ->(batch_size * num_heads , seq_len , head_dim)
        # ->(batch_size * num_heads , seq_len , seq_len , head_dim)
        if self.dropout_rate > 0.0:
            attn = self.dropout(attn)
        attn_v = torch.bmm(attn, v)

        attn_v = attn_v.transpose(0, 1).contiguous().view(-1, batch_size, self.embedding_dim).transpose(0, 1)
        output = self.w_o(attn_v)

        return output, attn


class UserTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(UserTransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, q, k, v, user_poi_pref_val=None, user_poi_pref_mask=None):
        for layer in self.layers:
            q = layer(q, k, v, user_poi_pref_val, user_poi_pref_mask)
        if self.norm is not None:
            q = self.norm(q)
        return q


class UserTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=32, dropout=0.3, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False):
        super(UserTransformerEncoderLayer, self).__init__()

        self.self_attn = UserMultiheadAttention(d_model, nhead, dropout, batch_first)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first

    def forward(self, q, k, v, user_poi_pref_val=None, user_poi_pref_mask=None):

        if self.norm_first:
            x = q + self.dropout1(
                self.self_attn(self.norm1(q), self.norm1(k), self.norm1(v), user_poi_pref_val, user_poi_pref_mask)[0])
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        else:
            x = self.norm1(q + self.dropout1(
                self.self_attn(q, k, v, user_poi_pref_val, user_poi_pref_mask)[0]))
            x = self.norm2(x + self.dropout2(self.feed_forward(x)))

        return x


class UserMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.3, batch_first=False):
        super(UserMultiheadAttention, self).__init__()

        self.embedding_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        self.batch_first = batch_first

        torch.nn.init.xavier_normal_(self.w_q.weight)
        torch.nn.init.xavier_normal_(self.w_k.weight)
        torch.nn.init.xavier_normal_(self.w_v.weight)
        torch.nn.init.xavier_normal_(self.w_o.weight)

    def forward(self, q1, k1, v1, user_poi_pref_val=None, user_poi_pref_mask=None):
        # #   Q(U,D) K(U,20,D) V(U,20,D)
        # user_count = q1.shape[0]
        # q = self.w_q(q1).reshape(user_count, 1, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        # # (U,1,H,d)->(H,U,1,d)
        # k = self.w_k(k1).reshape(user_count, -1, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        # # (U,20,H,d)->(H,U,20,d)
        # v = self.w_v(v1).reshape(user_count, -1, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        #
        # attn = torch.sum(torch.mul(q, k), dim=-1) / math.sqrt(self.head_dim)
        # # (H,U,1,d)*(H,U,20,d)->(H,U,20)
        # if user_poi_pref_mask is not None:
        #     attn = attn + user_poi_pref_mask
        # attn = torch.softmax(attn, dim=-1)
        # if user_poi_pref_val is not None:
        #     attn = (attn + user_poi_pref_val)
        # attn = self.dropout(attn)
        # attn_v = torch.sum(torch.mul(attn.unsqueeze(-1), v), dim=-2).permute()
        # # (H,U,20,1)*(H,U,20,d)->(H,U,d)
        # output = self.w_o(attn_v)
        q = self.w_q(q1)
        k = self.w_k(k1)
        v = self.w_v(v1)

        attn = torch.sum(torch.mul(q.unsqueeze(-2), k), dim=-1) / math.sqrt(
            self.embedding_dim)  # (P,1,D)*(P,20,D)->(P,20)
        if user_poi_pref_mask is not None:
            attn = attn + user_poi_pref_mask
        attn = torch.softmax(attn, dim=-1)
        if user_poi_pref_val is not None:
            attn = (attn + user_poi_pref_val)
        attn = self.dropout(attn)
        attn_v = torch.sum(torch.mul(attn.unsqueeze(-1), v), dim=-2)  # (P,20,1)*(P,20,D)->(P,D)
        output = self.w_o(attn_v)
        return output, attn


class CatTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CatTransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, q, k, v, attn_mask=None):
        for layer in self.layers:
            q = layer(q, k, v, attn_mask)
        if self.norm is not None:
            q = self.norm(q)
        return q


class CatTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=32, dropout=0.3, layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False):
        super(CatTransformerEncoderLayer, self).__init__()

        self.self_attn = CatMultiheadAttention(d_model, nhead, dropout, batch_first)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first

    def forward(self, q, k, v, attn_mask=None):
        if self.norm_first:
            x = q + self.dropout1(
                self.self_attn(self.norm1(q), self.norm1(k), self.norm1(v), attn_mask)[0])
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        else:
            x = self.norm1(q + self.dropout1(
                self.self_attn(q, k, v, attn_mask)[0]))
            x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x


class CatMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.3, batch_first=False):
        super(CatMultiheadAttention, self).__init__()

        self.embedding_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        self.batch_first = batch_first

        torch.nn.init.xavier_normal_(self.w_q.weight)
        torch.nn.init.xavier_normal_(self.w_k.weight)
        torch.nn.init.xavier_normal_(self.w_v.weight)
        torch.nn.init.xavier_normal_(self.w_o.weight)

    def forward(self, query, key, value, attn_mask=None):

        batch_size = query.shape[0]
        seq_len = query.shape[1]
        q = self.w_q(query).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).reshape(
            batch_size * self.num_heads, -1, self.head_dim)
        k = self.w_k(key).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).reshape(
            batch_size * self.num_heads, -1, self.head_dim)
        v = self.w_v(value).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).reshape(
            batch_size * self.num_heads, -1, self.head_dim)

        # (batch_size * num_heads, seq_len, seq_len)
        if attn_mask is not None:
            # out = input + (batch1 @ batch2)
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        else:
            attn = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn_v = torch.bmm(attn, v)

        attn_v = attn_v.transpose(0, 1).contiguous().view(-1, batch_size, self.embedding_dim).transpose(0, 1)
        output = self.w_o(attn_v)

        return output, attn


class FeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout_rate=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(embed_dim, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x


class TransformerModel(nn.Module):
    def __init__(self, user_count, poi_count, cat_count,
                 poi_transition,
                 poi_tran_neighbor,
                 poi_tran_neighbor_mask,

                 poi_distance,
                 poi_dis_neighbor,
                 poi_dis_neighbor_mask,

                 poi_transition_bias,

                 cat_transition,
                 cat_poi_mask,

                 user_poi_pref_idx,
                 user_poi_pref_val,
                 user_poi_pref_mask,
                 ):
        super(TransformerModel, self).__init__()
        self.user_count = user_count
        self.poi_count = poi_count
        self.cat_count = cat_count

        self.poi_transition = sparse_matrix_to_tensor(poi_transition).to(args.device)
        self.poi_tran_neighbor = poi_tran_neighbor
        self.poi_tran_neighbor_mask = poi_tran_neighbor_mask

        self.poi_transition_bias = poi_transition_bias

        self.poi_distance = sparse_matrix_to_tensor(poi_distance).to(args.device)
        self.poi_dis_neighbor = poi_dis_neighbor
        self.poi_dis_neighbor_mask = poi_dis_neighbor_mask
        #
        self.cat_transition = sparse_matrix_to_tensor(cat_transition).to(args.device)
        self.cat_poi_mask = cat_poi_mask

        self.user_poi_pref_idx = user_poi_pref_idx
        self.user_poi_pref_val = user_poi_pref_val
        self.user_poi_pref_mask = user_poi_pref_mask

        self.user_embedding = nn.Embedding(user_count, args.emb_dim)
        self.poi_embedding = nn.Embedding(poi_count, args.emb_dim)
        self.cat_embedding = nn.Embedding(cat_count, args.emb_dim)
        self.time_slot_embedding = TimeEncode(args.emb_dim)

        user_layer = UserTransformerEncoderLayer(d_model=args.emb_dim, nhead=args.num_heads, batch_first=True,
                                                 norm_first=True, dim_feedforward=256)
        self.user_encoder = UserTransformerEncoder(user_layer, num_layers=args.num_layers)

        cat_layer = CatTransformerEncoderLayer(d_model=args.emb_dim, nhead=args.num_heads, batch_first=True,
                                               norm_first=True, dim_feedforward=256)
        self.cat_encoder = CatTransformerEncoder(cat_layer, num_layers=args.num_layers)
        self.cat_tran_gcn = nn.ModuleList()
        for i in range(len(args.cat_gcn_dim) - 1):
            cat_tran_gcn_layer = GCNConv(args.cat_gcn_dim[i], args.cat_gcn_dim[i + 1])
            self.cat_tran_gcn.append(cat_tran_gcn_layer)
        self.cat_tran_leaky_relu = nn.LeakyReLU()
        # %===================================
        self.poi_tran_gcn = nn.ModuleList()
        for i in range(len(args.poi_gcn_dim) - 1):
            poi_tran_gcn_layer = GCNConv(args.poi_gcn_dim[i], args.poi_gcn_dim[i + 1])
            self.poi_tran_gcn.append(poi_tran_gcn_layer)
        self.poi_tran_leaky_relu = nn.LeakyReLU()
        #
        self.poi_dis_gcn = nn.ModuleList()
        for i in range(len(args.poi_gcn_dim) - 1):
            poi_dis_gcn_layer = GCNConv(args.poi_gcn_dim[i], args.poi_gcn_dim[i + 1])
            self.poi_dis_gcn.append(poi_dis_gcn_layer)
        self.poi_dis_leaky_relu = nn.LeakyReLU()

        sub_poi_t_layer = SubGraphTransformerEncoderLayer(d_model=args.emb_dim, nhead=args.num_heads, norm_first=True,
                                                          dim_feedforward=256)
        self.sub_poi_t_encoder = SubGraphTransformerEncoder(sub_poi_t_layer, num_layers=args.num_layers)
        sub_poi_d_layer = SubGraphTransformerEncoderLayer(d_model=args.emb_dim, nhead=args.num_heads, norm_first=True,
                                                          dim_feedforward=256)
        self.sub_poi_d_encoder = SubGraphTransformerEncoder(sub_poi_d_layer, num_layers=args.num_layers)
        self.fuse_t_d = nn.Linear(args.emb_dim*2, args.emb_dim)
        # %===================================
        seq_layer = SeqTransformerEncoderLayer(d_model=args.emb_dim * 4, nhead=args.num_heads,
                                               batch_first=True, norm_first=True, dim_feedforward=args.emb_dim * 8)
        self.seq_encoder = SeqTransformerEncoder(seq_layer, num_layers=args.num_seq_layers)

        self.fc_poi = nn.Linear(args.emb_dim * 4, poi_count)
        self.fc_cat = nn.Linear(args.emb_dim * 4, cat_count)

        torch.nn.init.xavier_normal_(self.fc_poi.weight)
        torch.nn.init.xavier_normal_(self.fc_cat.weight)

    def forward(self, user, tra, tra_cat, time_slot, attn_mask):
        # B batch_size; S seq_len; U user_cnt; P poi_cnt; C cat_cnt

        user = user.to(args.device)  # (B,S)
        tra = tra.to(args.device)  # (B,S)
        tra_cat = tra_cat.to(args.device)
        time_slot = time_slot.to(args.device)  # (B,S)
        attn_mask = attn_mask.to(args.device)
        # %% ===========================================================================================================

        out_poi_t_emb = self.poi_embedding.weight  # (128)
        out_poi_d_emb = self.poi_embedding.weight  # (128)
        all_poi_emb = self.poi_embedding.weight
        #
        for i in range(len(args.poi_gcn_dim) - 1):
            out_poi_t_emb = self.poi_tran_leaky_relu(self.poi_tran_gcn[i](out_poi_t_emb, self.poi_transition))

        for i in range(len(args.poi_gcn_dim) - 1):
            out_poi_d_emb = self.poi_dis_leaky_relu(self.poi_dis_gcn[i](out_poi_d_emb, self.poi_distance))
        #
        out_poi_t_emb = self.sub_poi_t_encoder(out_poi_t_emb,
                                               out_poi_t_emb[self.poi_tran_neighbor],
                                               all_poi_emb[self.poi_tran_neighbor],
                                               mask=self.poi_tran_neighbor_mask,
                                               bias=self.poi_transition_bias)
        out_poi_d_emb = self.sub_poi_d_encoder(out_poi_d_emb,
                                               out_poi_d_emb[self.poi_dis_neighbor],
                                               all_poi_emb[self.poi_dis_neighbor],
                                               mask=self.poi_dis_neighbor_mask)
        out_poi_emb = self.fuse_t_d(torch.cat((out_poi_t_emb, out_poi_d_emb), dim=-1))

        # %% ===========================================================================================================
        all_user_emb = self.user_embedding.weight
        out_user_emb = self.user_encoder(all_user_emb,
                                         out_poi_emb[self.user_poi_pref_idx],
                                         out_poi_emb[self.user_poi_pref_idx],
                                         user_poi_pref_val=self.user_poi_pref_val,
                                         user_poi_pref_mask=self.user_poi_pref_mask).to(args.device)

        # %% ===========================================================================================================
        all_cat_emb = self.cat_embedding.weight
        out_cat_emb = self.cat_encoder(all_cat_emb.unsqueeze(0),
                                       out_poi_emb.unsqueeze(0),
                                       out_poi_emb.unsqueeze(0),
                                       attn_mask=self.cat_poi_mask)
        out_cat_emb = out_cat_emb.squeeze()
        for i in range(len(args.cat_gcn_dim) - 1):
            out_cat_emb = self.cat_tran_leaky_relu(self.cat_tran_gcn[i](out_cat_emb, self.cat_transition))
        # %% ===========================================================================================================

        tra_emb = out_poi_emb[tra]
        tra_cat_emb = out_cat_emb[tra_cat]
        user_emb = out_user_emb[user]
        time_slot_emb = self.time_slot_embedding(time_slot)
        input_emb = torch.cat([tra_emb, tra_cat_emb, user_emb, time_slot_emb], dim=-1)

        out = self.seq_encoder(input_emb, attn_mask=attn_mask)
        y_cat = self.fc_cat(out)
        y_pre = self.fc_poi(out)
        return y_pre, y_cat
