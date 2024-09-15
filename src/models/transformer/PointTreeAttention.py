from imp import C_EXTENSION
from termios import B110
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from einops.einops import rearrange
import time
import math
from timm.models.layers import DropPath
import torch_scatter
import matplotlib.pyplot as plt
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
from lib.pointops2.functions import pointops
from torch_scatter import scatter_softmax

ONE_BATCH = True


class MLP_VSA_Layer(nn.Module):
    def __init__(self, dim, n_latents=8):
        super(MLP_VSA_Layer, self).__init__()
        self.dim = dim

        self.pre_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
        )
        self.score = nn.Linear(dim, n_latents)

        conv_dim = dim * 8
        self.conv_dim = conv_dim
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 1, 1, bias=False),
        )
        self.norm = nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01)
        self.mhsa = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, inp, inverse, topk_idx):
        x = self.pre_mlp(inp)
        attn = torch_scatter.scatter_softmax(self.score(x), inverse, dim=0)
        dot = (attn[:, :, None] * x.view(-1, 1, self.dim)).view(-1, self.dim * 8)
        x_ = torch_scatter.scatter_sum(dot, inverse, dim=0)
        h = x_[topk_idx[0]]
        hs = self.norm(h.view(-1, self.dim)).view(-1, 8, self.dim)
        hs = self.mhsa(x.view(-1, 1, self.dim), hs, hs)[0]
        hs = hs.view(-1, self.dim)
        return torch.cat([inp, hs], dim=-1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PTA(nn.Module):
    def __init__(
            self,
            nhead,
            in_dim,
            dim,
            topks=[4, 4, 4, 4],
            knn_k=8,
            scale=None,
            use_dropout=False,
            attention_dropout=0.1,
            qkv_bias=False,
            mlp_ratio=1.,
            act_layer=nn.GELU,
            drop=0.,
            norm_layer=nn.LayerNorm,
            attn_drop=0.,
            drop_path=0.,
            proj_drop=0.,
            num_points=20
    ):
        super().__init__()
        self.use_dropout = use_dropout
        self.topks = topks
        self.nhead = nhead
        self.knn_k = knn_k
        self.scale = (dim // nhead) ** -0.5

        self.dim = dim
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.in_dim = in_dim
        self.qkv = nn.Linear(in_dim, dim * 3, bias=qkv_bias)
        self.mlp_fine = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim,
                            act_layer=act_layer, drop=drop)
        self.norm1 = norm_layer(in_dim)
        self.norm2 = norm_layer(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_points = num_points
        self.fc = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=in_dim, act_layer=act_layer,
                      drop=drop)
        self.latlayer = nn.Linear(dim, dim)

    def process_coarse_level(self, query, key, value, topk, key_padding_mask, query_padding_mask):
        B, M, H, C = query.shape
        B, N, H, C = key.shape
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        softmax_temp = 1.0 / (C ** 0.5)  # sqrt(D)

        if ONE_BATCH == False:
            mask_mat = key_padding_mask.unsqueeze(-2).repeat(1, M, 1)
            mask_mat = mask_mat.unsqueeze(-1).repeat(1, 1, 1, H)
            QK[mask_mat] = -1e9

        A = torch.softmax(softmax_temp * QK, dim=-2)

        A_mean = torch.mean(A, dim=-1)

        _, topk_idx = torch.topk(A_mean, dim=-1, k=topk, largest=True)
        A = self.attn_drop(A)
        message = torch.einsum("nlsh,nshd->nlhd", A, value).reshape(B, N,
                                                                    H * C)  # .reshape(bs, h, w, self.nhead, cur_dim)
        message = self.proj(message)
        message = self.proj_drop(message)
        value = value.reshape(B, N, H * C)
        message = value.squeeze(1) + message

        if ONE_BATCH == True:
            message = message.squeeze(0)
            topk_idx = topk_idx.squeeze(0)
        else:
            message = message[~query_padding_mask, :]
            topk_idx = topk_idx[~query_padding_mask, :]

        return A, message, topk_idx

    def process_fine_full_level(self, query, key, value, topk, key_padding_mask, query_padding_mask):
        B, M, H, C = query.shape
        B, N, H, C = key.shape
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        softmax_temp = 1.0 / (C ** 0.5)  # sqrt(D)
        mask_mat = key_padding_mask.unsqueeze(-2).repeat(1, M, 1)
        mask_mat = mask_mat.unsqueeze(-1).repeat(1, 1, 1, H)

        QK[mask_mat] = -1e9
        A = torch.softmax(softmax_temp * QK, dim=-2)
        A = self.attn_drop(A)
        message = torch.einsum("nlsh,nshd->nlhd", A, value).reshape(B, N,
                                                                    H * C)  # .reshape(bs, h, w, self.nhead, cur_dim)
        message = self.proj_fine(message)
        message = self.proj_drop(message)
        value = value.reshape(B, N, H * C)
        message = value.squeeze(1) + message

        message = message[~query_padding_mask, :]

        return A, message

    def process_m_level(self, query, key, value, topk_score, index_query, index_kv, kv_topk_mask, kv_topk_idx,
                        topk_prev, topk, inverse, index_prev):
        topk_idx = None

        H, N, C = key.shape
        H, M, C = query.shape

        index_query_counts = index_query.bincount()
        n_max = index_query_counts.max()
        index_query_counts = index_query_counts.cumsum(dim=-1)  # [N]
        index_query_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_query_counts], 0)

        query = query * self.scale

        attn_flat = pointops.attention_step1_v2(query.reshape(M * H, 1, C).float(), key.reshape(N * H, 1, C).float(),
                                                index_kv.int(), index_query_offsets.int(), n_max)

        relative_position_bias = 0.0

        attn_flat = attn_flat + relative_position_bias

        attn_flat[kv_topk_mask] = -1e9

        softmax_attn = attn_flat.reshape(H * M * topk_prev, -1)
        softmax_attn_flat = torch.softmax(softmax_attn, dim=-1).reshape(-1, 1)

        softmax_attn_flat = self.attn_drop(softmax_attn_flat)

        softmax_attn_flat = softmax_attn_flat * topk_score.unsqueeze(-1)

        message = pointops.attention_step2(softmax_attn_flat.float(), value.reshape(N * H, 1, C).float(),
                                           index_query.int(), index_kv.int())
        message = message.reshape(H, M, C).permute(1, 0, 2).reshape(M, H * C)
        message = self.proj(message)
        message = self.proj_drop(message)
        value = value.permute(1, 0, 2).reshape(N, H * C)
        message = value.squeeze(1) + message
        attn_matrix = softmax_attn_flat.reshape(H * M, -1)

        topk_score, topk_idx = torch.topk(attn_matrix, dim=-1, k=topk, largest=True)
        topk_idx = (index_query_offsets[:-1].unsqueeze(-1) + topk_idx).reshape(-1, 1)

        topk_idx = torch.gather(kv_topk_idx.unsqueeze(-1), index=topk_idx, dim=0).reshape(H, M, topk).permute(1, 2, 0)
        topk_score = topk_score.reshape(H, M, topk).permute(1, 2, 0)
        return softmax_attn_flat, message, topk_score, topk_idx, None

    def process_fine_level(self, query, key, value, index_query, index_kv, topk, final):
        topk_idx = None
        device = 'cuda'

        M, H, C = query.shape
        N, H, C = key.shape

        index_query_counts = index_query.bincount()
        n_max = index_query_counts.max()
        index_query_counts = index_query_counts.cumsum(dim=-1)  # [N]
        index_query_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_query_counts], 0)

        query = query * self.scale

        attn_flat = pointops.attention_step1_v2(query.float(), key.contiguous().float(), index_kv.int(),
                                                index_query_offsets.int(), n_max)

        relative_position_bias = 0.0

        attn_flat = attn_flat + relative_position_bias

        softmax_attn_flat = scatter_softmax(src=attn_flat, index=index_query, dim=0)

        if not final:
            st = time.time()
            softmax_attn_flat_sort = torch.mean(softmax_attn_flat, dim=-1)
            softmax_attn_flat_sort = softmax_attn_flat_sort - index_query
            _, indices = torch.sort(softmax_attn_flat_sort, dim=-1, descending=True)
            index_topk = index_query_offsets[:-1].reshape(-1, 1) + torch.arange(topk, device=device).reshape(1, -1)
            topk_idx = indices[index_topk].reshape(-1)
            topk_idx = torch.gather(index_kv, index=topk_idx, dim=0).reshape(M, topk)

        softmax_attn_flat = self.attn_drop(softmax_attn_flat)

        message = pointops.attention_step2(softmax_attn_flat.float(), value.contiguous().float(), index_query.int(),
                                           index_kv.int())
        message = message.reshape(M, H * C)
        message = self.proj(message)
        message = self.proj_drop(message)
        value = value.reshape(N, H * C)
        message = value + message
        return softmax_attn_flat, message, topk_idx

    def forward(self, points, indices, inverses, slens, counts):
        topk = self.topks[0]
        indices_new = []
        messages = []
        for i, (point, index, inverse, slen, count) in enumerate(
                zip(reversed(points), reversed(indices), reversed(inverses), reversed(slens), reversed(counts))):
            point = self.norm1(point)
            topk = self.topks[i]
            device = point.device
            if i == 0:
                if ONE_BATCH == True:
                    point = point.unsqueeze(0)
                    padding_mask = None
                else:
                    separate = torch.split(point, list(slen), dim=0)
                    point, padding_mask, _ = pad_sequence(separate, require_padding_mask=True, batch_first=True)
                B, N, C = point.shape
                if 2 * topk >= N:
                    topk = int(N//2)
                qkv = self.qkv(point).reshape(B, N, 3, self.nhead, self.dim // self.nhead).permute(2, 0, 1, 3, 4)
                query, key, value = qkv[0], qkv[1], qkv[2]
                st = time.time()
                A_coarse, final_message, topk_idx = self.process_coarse_level(
                    query, key, value, topk, padding_mask, padding_mask
                )  # Full attention for coarest level
                final_message = self.norm2(final_message)
            else:
                N, C = point.shape
                message_prev = final_message[inverse_prev]

                qkv = self.qkv(point + self.fc(message_prev)).reshape(N, 3, self.nhead,
                                                                       self.dim // self.nhead).permute(1, 0, 2, 3)
                query, key, value = qkv[0], qkv[1], qkv[2]
                M, H, C = query.shape
                N, H, C = key.shape

                _, k = index_prev.shape
                N_prev, K = topk_idx.shape
                index_mask = torch.arange(k, device=device).unsqueeze(0) < count_prev.unsqueeze(-1)

                index_prev_mask = torch.stack([index_prev, index_mask], dim=0)
                origin_index_mask = index_prev_mask[:, topk_idx]

                kv_topk_idx_mask = origin_index_mask[:, inverse_prev, :, :].transpose(-1, -2).reshape(2, M, -1)

                kv_topk_idx, kv_topk_mask = kv_topk_idx_mask[0], kv_topk_idx_mask[1].to(torch.bool)

                index_0 = torch.arange(M, device=device).reshape(M, 1).repeat(1, k * K)
                topk_info = torch.stack([index_0, kv_topk_idx], dim=0)
                topk_info = topk_info[:, kv_topk_mask]
                index_0, index_1 = topk_info

                final = True if i == len(points) - 1 else False

                A_fine, final_message, topk_idx = self.process_fine_level(
                    query, key, value, index_0, index_1, topk, final
                )  # tree attention
                final_message = self.norm2(final_message)
                final_message = final_message + self.drop_path(self.mlp_fine(final_message))

                final_message = final_message + message_prev
            inverse_prev = inverse
            index_prev = index
            count_prev = count
            topk_prev = topk
            messages.append(final_message)
        return final_message, A_coarse, list(reversed(indices_new)), messages[:-1]


class PTA_cross(nn.Module):
    def __init__(
            self,
            nhead,
            in_dim,
            dim,
            topks=[4, 4, 4, 4],
            knn_k=8,
            scale=None,
            use_dropout=False,
            attention_dropout=0.1,
            qkv_bias=False,
            mlp_ratio=1.,
            act_layer=nn.GELU,
            drop=0.,
            norm_layer=nn.LayerNorm,
            attn_drop=0.,
            drop_path=0.,
            proj_drop=0.,
            num_points=20
    ):
        super().__init__()
        self.use_dropout = use_dropout
        self.topks = topks
        self.nhead = nhead
        self.knn_k = knn_k
        self.scale = (dim // nhead) ** -0.5

        self.dim = dim
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.in_dim = in_dim
        self.kv = nn.Linear(in_dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(in_dim, dim * 2, bias=qkv_bias)
        self.mlp_fine = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim,
                            act_layer=act_layer, drop=drop)
        self.norm1 = norm_layer(in_dim)
        self.norm2 = norm_layer(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_points = num_points
        self.fc = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=in_dim, act_layer=act_layer,
                      drop=drop)
        self.latlayer = nn.Linear(dim, dim)

    def process_coarse_level(self, query, key, value, q_value, topk):
        B, M, H, C = query.shape
        B, N, H, C = key.shape
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        softmax_temp = 1.0 / (C ** 0.5)  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)

        A_mean = torch.mean(A, dim=-1)

        _, topk_idx = torch.topk(A_mean, dim=-1, k=topk, largest=True)
        A = self.attn_drop(A)
        message = torch.einsum("nlsh,nshd->nlhd", A, value).reshape(B, M,
                                                                    H * C)  # .reshape(bs, h, w, self.nhead, cur_dim)
        message = self.proj(message)
        message = self.proj_drop(message)
        q_value = q_value.reshape(B, M, H * C)
        message = q_value + message

        message = message.squeeze(0)
        topk_idx = topk_idx.squeeze(0)

        return A, message, topk_idx

    def process_fine_full_level(self, query, key, value, topk, key_padding_mask, query_padding_mask):
        B, M, H, C = query.shape
        B, N, H, C = key.shape
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        judge = torch.isnan(QK)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        softmax_temp = 1.0 / (C ** 0.5)  # sqrt(D)
        mask_mat = key_padding_mask.unsqueeze(-2).repeat(1, M, 1)
        mask_mat = mask_mat.unsqueeze(-1).repeat(1, 1, 1, H)

        QK[mask_mat] = -1e9
        A = torch.softmax(softmax_temp * QK, dim=-2)
        judge = torch.isnan(A)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        A = self.attn_drop(A)
        message = torch.einsum("nlsh,nshd->nlhd", A, value).reshape(B, N,
                                                                    H * C)  # .reshape(bs, h, w, self.nhead, cur_dim)
        message = self.proj_fine(message)
        message = self.proj_drop(message)
        value = value.reshape(B, N, H * C)
        message = value.squeeze(1) + message

        judge = torch.isnan(message)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        message = message[~query_padding_mask, :]

        return A, message

    def process_m_level(self, query, key, value, topk_score, index_query, index_kv, kv_topk_mask, kv_topk_idx,
                        topk_prev, topk, inverse, index_prev):
        topk_idx = None

        H, N, C = key.shape
        H, M, C = query.shape

        index_query_counts = index_query.bincount()
        n_max = index_query_counts.max()
        index_query_counts = index_query_counts.cumsum(dim=-1)  # [N]
        index_query_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_query_counts], 0)

        query = query * self.scale

        attn_flat = pointops.attention_step1_v2(query.reshape(M * H, 1, C).float(), key.reshape(N * H, 1, C).float(),
                                                index_kv.int(), index_query_offsets.int(), n_max)

        relative_position_bias = 0.0

        attn_flat = attn_flat + relative_position_bias

        attn_flat[kv_topk_mask] = -1e9

        softmax_attn = attn_flat.reshape(H * M * topk_prev, -1)
        softmax_attn_flat = torch.softmax(softmax_attn, dim=-1).reshape(-1, 1)

        softmax_attn_flat = self.attn_drop(softmax_attn_flat)

        softmax_attn_flat = softmax_attn_flat * topk_score.unsqueeze(-1)

        message = pointops.attention_step2(softmax_attn_flat.float(), value.reshape(N * H, 1, C).float(),
                                           index_query.int(), index_kv.int())
        message = message.reshape(H, M, C).permute(1, 0, 2).reshape(M, H * C)
        message = self.proj(message)
        message = self.proj_drop(message)
        value = value.permute(1, 0, 2).reshape(N, H * C)
        message = value.squeeze(1) + message
        judge = torch.isnan(message)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True
        attn_matrix = softmax_attn_flat.reshape(H * M, -1)

        topk_score, topk_idx = torch.topk(attn_matrix, dim=-1, k=topk, largest=True)
        topk_idx = (index_query_offsets[:-1].unsqueeze(-1) + topk_idx).reshape(-1, 1)

        topk_idx = torch.gather(kv_topk_idx.unsqueeze(-1), index=topk_idx, dim=0).reshape(H, M, topk).permute(1, 2, 0)
        topk_score = topk_score.reshape(H, M, topk).permute(1, 2, 0)
        return softmax_attn_flat, message, topk_score, topk_idx, None

    def process_fine_level(self, query, key, value, q_value, index_query, index_kv, index_query_counts, topk, final):

        M, H, C = query.shape
        N, H, C = key.shape

        topk_idx = None

        device = query.device
        n_max = index_query_counts.max()
        index_query_counts = index_query_counts.cumsum(dim=-1)  # [N]
        index_query_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), index_query_counts], 0)

        query = query * self.scale

        if N > M:
            padded = torch.zeros(N - M, H, C, device=device)
            query = torch.cat([query, padded], dim=0)
            prev_len = index_kv.shape[0]
            padded = torch.zeros(N - M, topk, device=device).reshape(-1)
            index_kv_ = torch.cat([index_kv, padded], dim=0)
            last = index_query_offsets[-1]
            padded = torch.arange(1, (N - M) + 1, device=device) * topk
            padded = last + padded
            index_query_offsets_ = torch.cat([index_query_offsets, padded], dim=0)
            attn_flat = pointops.attention_step1_v2(query.float(), key.contiguous().float(), index_kv_.int(),
                                                    index_query_offsets_.int(), n_max)
            attn_flat = attn_flat[:prev_len]
        elif M > N:
            padded = torch.zeros(M - N, H, C, device=device)
            key = torch.cat([key, padded], dim=0)
            value = torch.cat([value, padded], dim=0)
            attn_flat = pointops.attention_step1_v2(query.float(), key.contiguous().float(), index_kv.int(),
                                                    index_query_offsets.int(), n_max)
        else:
            attn_flat = pointops.attention_step1_v2(query.float(), key.contiguous().float(), index_kv.int(),
                                                    index_query_offsets.int(), n_max)

        relative_position_bias = 0.0

        attn_flat = attn_flat + relative_position_bias

        softmax_attn_flat = scatter_softmax(src=attn_flat, index=index_query, dim=0)

        softmax_attn_flat = self.attn_drop(softmax_attn_flat)

        if not final:
            st = time.time()
            softmax_attn_flat_sort = torch.mean(softmax_attn_flat, dim=-1)
            softmax_attn_flat_sort = softmax_attn_flat_sort - index_query
            _, indices = torch.sort(softmax_attn_flat_sort, dim=-1, descending=True)
            index_topk = index_query_offsets[:-1].reshape(-1, 1) + torch.arange(topk, device=device).reshape(1, -1)
            topk_idx = indices[index_topk].reshape(-1)
            topk_idx = torch.gather(index_kv, index=topk_idx, dim=0).reshape(M, topk)

        message = pointops.attention_step2(softmax_attn_flat.float(), value.contiguous().float(), index_query.int(),
                                           index_kv.int())
        message = message.reshape(M, H * C)
        message = self.proj(message)
        message = self.proj_drop(message)
        q_value = q_value.reshape(M, H * C)
        message = q_value + message
        return softmax_attn_flat, message, topk_idx

    def coarse_forward(self, point_q, point_kv, topk):
        B, M, C = point_q.shape
        B, N, C = point_kv.shape
        kv = self.kv(point_kv).reshape(B, N, 2, self.nhead, self.dim // self.nhead).permute(2, 0, 1, 3, 4)
        key, value = kv[0], kv[1]
        q = self.q(point_q).reshape(B, M, 2, self.nhead, self.dim // self.nhead).permute(2, 0, 1, 3, 4)
        query, q_value = q[0], q[1]
        A_coarse, final_message, topk_idx = self.process_coarse_level(
            query, key, value, q_value, topk
        )  # Full attention for coarest level

        final_message = self.norm2(final_message)
        return A_coarse, final_message, topk_idx

    def fine_forward(self, point_q, point_kv, topk,
                     inverse_prev, index_prev, count_prev, topk_idx, final):
        device = point_q.device
        M, C = point_q.shape
        N, C = point_kv.shape
        kv = self.kv(point_kv).reshape(N, 2, self.nhead, self.dim // self.nhead).permute(1, 0, 2, 3)
        key, value = kv[0], kv[1]
        q = self.q(point_q).reshape(M, 2, self.nhead, self.dim // self.nhead).permute(1, 0, 2, 3)
        query, q_value = q[0], q[1]
        _, k = index_prev.shape
        N_prev, K = topk_idx.shape

        index_mask = torch.arange(k, device=device).unsqueeze(0) < count_prev.unsqueeze(-1)

        index_prev_mask = torch.stack([index_prev, index_mask], dim=0)
        origin_index_mask = index_prev_mask[:, topk_idx]

        kv_topk_idx_mask = origin_index_mask[:, inverse_prev, :, :].transpose(-1, -2).reshape(2, M, -1)

        kv_topk_idx, kv_topk_mask = kv_topk_idx_mask[0], kv_topk_idx_mask[1].to(torch.bool)

        index_0 = torch.arange(M, device=device).reshape(M, 1).repeat(1, k * K)
        topk_info = torch.stack([index_0, kv_topk_idx], dim=0)
        topk_info = topk_info[:, kv_topk_mask]
        index_0, index_1 = topk_info

        index_0_counts = torch.sum(kv_topk_mask, dim=-1)

        A_fine, final_message, topk_idx = self.process_fine_level(
            query, key, value, q_value, index_0, index_1, index_0_counts, topk, final
        )  # tree attention
        final_message = self.norm2(final_message)
        final_message = final_message + self.drop_path(self.mlp_fine(final_message))
        return A_fine, final_message, topk_idx

    def forward(self, points_src, points_tgt, indices_src, indices_tgt, inverses_src, inverses_tgt, counts_src,
                counts_tgt):
        src_messages, tgt_messages = [], []
        for i, (point_src, point_tgt, index_src, index_tgt, inverse_src, inverse_tgt, count_src, count_tgt) in \
                enumerate(zip(reversed(points_src), reversed(points_tgt), reversed(indices_src), reversed(indices_tgt),
                              reversed(inverses_src), reversed(inverses_tgt), reversed(counts_src),
                              reversed(counts_tgt))):
            point_src = self.norm1(point_src)
            point_tgt = self.norm1(point_tgt)
            topk = self.topks[i]
            device = point_src.device
            N_src = point_src.shape[0]
            N_tgt = point_tgt.shape[0]
            if topk >= min(N_src, N_tgt):
                topk = min(N_src, N_tgt)
            if i == 0:
                point_src = point_src.unsqueeze(0)
                point_tgt = point_tgt.unsqueeze(0)
                src_A_coarse, src_message, src_topk_idx = self.coarse_forward(point_src, point_tgt, topk)
                tgt_A_coarse, tgt_message, tgt_topk_idx = self.coarse_forward(point_tgt, point_src, topk)
            else:
                src_message_prev = src_message[inverse_prev_src]
                tgt_message_prev = tgt_message[inverse_prev_tgt]
                point_src = point_src + self.fc(src_message_prev)
                point_tgt = point_tgt + self.fc(tgt_message_prev)
                final = True if i == len(points_src) - 1 else False
                src_A_fine, src_message, src_topk_idx = self.fine_forward(point_src, point_tgt, topk,
                                                                          inverse_prev_src, index_prev_tgt,
                                                                          count_prev_tgt, src_topk_idx, final)
                tgt_A_fine, tgt_message, tgt_topk_idx = self.fine_forward(point_tgt, point_src, topk,
                                                                          inverse_prev_tgt, index_prev_src,
                                                                          count_prev_src, tgt_topk_idx, final)
                src_message = src_message + src_message_prev
                tgt_message = tgt_message + tgt_message_prev
            inverse_prev_src = inverse_src
            index_prev_src = index_src
            count_prev_src = count_src
            inverse_prev_tgt = inverse_tgt
            index_prev_tgt = index_tgt
            count_prev_tgt = count_tgt
            src_messages.append(src_message)
            tgt_messages.append(tgt_message)
        return src_message, tgt_message, src_A_coarse, tgt_A_coarse, src_messages[:-1], tgt_messages[:-1]
