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
# from pcdet.ops.votr_ops import votr_utils
import matplotlib.pyplot as plt
# import spconv.pytorch as spconv
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

        # the learnable latent codes can be obsorbed by the linear projection
        self.score = nn.Linear(dim, n_latents)

        conv_dim = dim * 8
        self.conv_dim = conv_dim

        # conv ffn
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            # nn.Conv2d(conv_dim, conv_dim, 3, 1, dilation=2, padding=2, groups=conv_dim, bias=False),
            # nn.BatchNorm2d(conv_dim),
            # nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 1, 1, bias=False),
        )

        # decoder
        self.norm = nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01)
        self.mhsa = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, inp, inverse, topk_idx):
        x = self.pre_mlp(inp)

        # encoder
        # instead of matmul(x, lantent_code), map attention matrix by self.score
        attn = torch_scatter.scatter_softmax(self.score(x), inverse, dim=0)
        # attn.shape = [N,C,1], attn[:,:,None].shape = [N,C,1]
        dot = (attn[:, :, None] * x.view(-1, 1, self.dim)).view(-1, self.dim * 8)
        # voxel features
        x_ = torch_scatter.scatter_sum(dot, inverse, dim=0)

        # conv ffn
        # batch_size = int(coords[:, 0].max() + 1)
        # h = spconv.SparseConvTensor(F.relu(x_), coords.int(), bev_shape, batch_size).dense().squeeze(-1)
        # h = self.conv_ffn(h).permute(0,2,3,1).contiguous().view(-1, self.conv_dim)
        # flatten_indices = coords[:, 0] * bev_shape[0] * bev_shape[1] + coords[:, 1] * bev_shape[1] + coords[:, 2]
        # h = h[flatten_indices.long(), :]
        # h = h[inverse, :]
        h = x_[topk_idx[0]]

        # decoder
        hs = self.norm(h.view(-1, self.dim)).view(-1, 8, self.dim)
        hs = self.mhsa(x.view(-1, 1, self.dim), hs, hs)[0]
        hs = hs.view(-1, self.dim)

        # skip connection
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


class QTAttA(nn.Module):
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
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm1 = norm_layer(in_dim)
        self.norm2 = norm_layer(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_points = num_points
        self.fc = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=in_dim, act_layer=act_layer,
                      drop=drop)

    def process_coarse_level(self, query, key, value, topk, key_padding_mask, query_padding_mask):
        # Correct
        B, M, H, C = query.shape
        B, N, H, C = key.shape

        # B,Number_Queries,Number_Keys,H
        # Correct
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

        # plt.imshow(A.cpu().numpy()[1,:,:,0])
        # plt.colorbar()
        # plt.show()

        judge = torch.isnan(A)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)

        mask = torch.ones_like(A)
        # Correct
        mask = mask.scatter(dim=-2, index=topk_idx, src=torch.zeros_like(topk_idx).float())

        A = self.attn_drop(A)
        message = torch.einsum("nlsh,nshd->nlhd", A * mask, value).reshape(B, N,
                                                                           H * C)  # .reshape(bs, h, w, self.nhead, cur_dim)
        message = self.proj(message)
        message = self.proj_drop(message)
        value = value.reshape(B, N, H * C)
        message = value.squeeze(1) + message

        judge = torch.isnan(message)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        topk_score = topk_score[~query_padding_mask, :]
        topk_idx = topk_idx[~query_padding_mask, :]
        message = message[~query_padding_mask, :]

        return A, message, topk_score, topk_idx

    def process_fine_level(self, query, key, value, topk_score, index_query, index_kv, index_softmax, kv_topk_idx,
                           topk_prev, topk, inverse, index_prev, final=False):
        topk_idx = None

        H, N, C = key.shape
        H, M, C = query.shape

        index_query_counts = index_query.bincount()
        # max number of keys
        n_max = index_query_counts.max()
        # offset
        index_query_counts = index_query_counts.cumsum(dim=-1)  # [N]

        # why N+1
        index_query_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), index_query_counts], 0)

        query = query * self.scale

        attn_flat = pointops.attention_step1_v2(query.reshape(M * H, 1, C).float(), key.reshape(N * H, 1, C).float(),
                                                index_kv.int(), index_query_offsets.int(), n_max)

        relative_position_bias = 0.0

        attn_flat = attn_flat + relative_position_bias

        softmax_attn_flat = scatter_softmax(src=attn_flat, index=index_softmax, dim=0)

        softmax_attn_flat = softmax_attn_flat * topk_score.unsqueeze(-1)

        softmax_attn_flat = self.attn_drop(softmax_attn_flat)

        if not final:
            #######################SLOW##############################
            query_offsets_ = index_query_counts.clone()
            query_offsets_[1:] = query_offsets_[1:] - query_offsets_[:-1]
            # H, M, N
            separate = torch.split(softmax_attn_flat, list(query_offsets_), dim=0)
            # padding_mask: masked locations are True
            attn_matrix, _, _ = pad_sequence(separate, require_padding_mask=False, batch_first=True)

            ####### TODO: wrong ##############
            # A_prev = torch_scatter.scatter_sum(attn_matrix.reshape(H,M,-1).permute(1,2,0), inverse, dim=0)
            # # M_prev, key, H
            # A_prev = torch.mean(A_prev,dim=-1)
            ##################################

            attn_matrix = attn_matrix.reshape(H * M, -1)

            topk_score, topk_idx = torch.topk(attn_matrix, dim=-1, k=topk, largest=True)

            # H,M,K -> H*M,K
            topk_idx = (index_query_offsets[:-1].unsqueeze(-1) + topk_idx).reshape(-1, 1)
            #######################SLOW##################################

            mask = torch.ones_like(softmax_attn_flat)
            mask = mask.scatter(dim=0, index=topk_idx, src=torch.zeros_like(topk_idx).float())
            # message is only computed within the unmasked
            message = pointops.attention_step2((softmax_attn_flat * mask).float(), value.reshape(N * H, 1, C).float(),
                                               index_query.int(), index_kv.int())
            message = message.reshape(H, M, C).permute(1, 0, 2).reshape(M, H * C)
            message = self.proj(message)
            message = self.proj_drop(message)
            value = value.permute(1, 0, 2).reshape(N, H * C)
            message = value.squeeze(1) + message
            # message = value_aggregation_op(A * mask, value.contiguous(), kv_topk_idx)
        else:
            message = pointops.attention_step2(softmax_attn_flat.float(), value.reshape(N * H, 1, C).float(),
                                               index_query.int(), index_kv.int())
            message = message.reshape(H, M, C).permute(1, 0, 2).reshape(M, H * C)
            message = self.proj(message)
            message = self.proj_drop(message)
            value = value.permute(1, 0, 2).reshape(N, H * C)
            message = value.squeeze(1) + message
            # message = value_aggregation_op(A, value.contiguous(), kv_topk_idx)

        judge = torch.isnan(message)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        # should be modified
        if not final:
            # idx: B,(N,4),4K,H
            # topk_idx: B,N,K_New,H
            topk_idx = torch.gather(kv_topk_idx.unsqueeze(-1), index=topk_idx, dim=0).reshape(H, M, topk).permute(1, 2,
                                                                                                                  0)
            # topk_idx_prev = torch.gather(index_kv, index=topk_idx_prev, dim=-2)

            # self_idx = torch.arange(topk_idx.shape[1]).reshape(1,-1,1,1).repeat(B,1,1,H)
            # topk_idx = torch.cat([topk_idx,self_idx],dim=-2)

            # topk_idx: B,h//2,w//2,2,2,K_New,H
            # topk_idx = rearrange(topk_idx, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2)  # reshape back
            # topk_score = rearrange(
            #     topk_score, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2
            # )  # reshape back
        return softmax_attn_flat, message, topk_score, topk_idx, None

    def forward(self, points, indices, inverses, slens, counts):
        """Multi-head quadtree attention
        Args:
            queries: Query pyramid [N, C, H, W]
            keys: Key pyramid [N, C, H, W]
            values: Value pyramid [N, C, H, W]
        Returns:
            message: (N, C, H, W)
        """

        topk = self.topks[0]
        indices_new = []
        for i, (point, index, inverse, slen, count) in enumerate(
                zip(reversed(points), reversed(indices), reversed(inverses), reversed(slens), reversed(counts))):
            point = self.norm1(point)
            topk = self.topks[i]
            device = point.device
            if i == 0:
                separate = torch.split(point, list(slen), dim=0)
                # padding_mask: masked locations are True
                point, padding_mask, _ = pad_sequence(separate, require_padding_mask=True, batch_first=True)
                B, N, C = point.shape
                if N < topk * 4:
                    topk = max(N // 4, 1)
                    print('N < 32 with', point.shape)
                qkv = self.qkv(point).reshape(B, N, 3, self.nhead, self.dim // self.nhead).permute(2, 0, 1, 3, 4)
                query, key, value = qkv[0], qkv[1], qkv[2]
                # topk_idx B,Number_Queries,K,H
                # Correct
                st = time.time()
                A_coarse, final_message, topk_score, topk_idx = self.process_coarse_level(
                    query, key, value, topk, padding_mask, padding_mask
                )  # Full attention for coarest level
                # print('coarse_time',time.time() - st)
            else:
                N, C = point.shape
                final_message = final_message[inverse_prev]
                judge = torch.isnan(final_message)
                judge_1 = torch.zeros_like(judge)
                result = torch.all(torch.eq(judge_1, judge))
                assert result == True
                qkv = self.qkv(point + self.fc(final_message)).reshape(N, 3, self.nhead,
                                                                       self.dim // self.nhead).permute(1, 2, 0, 3)
                query, key, value = qkv[0], qkv[1], qkv[2]
                H, M, C = query.shape
                H, N, C = key.shape
                judge = torch.isnan(qkv)
                judge_1 = torch.zeros_like(judge)
                result = torch.all(torch.eq(judge_1, judge))
                assert result == True
                N_prev, K, H = topk_idx.shape
                _, k = index_prev.shape
                index_mask = torch.arange(k, device=device).unsqueeze(0) < count_prev.unsqueeze(-1)
                # B, N_prev, knn_k, H
                # index_prev = index_prev.unsqueeze(-1).repeat(1,1,1,H)
                # transformed_idx = torch.zeros(B,N,K,H,k).to(points[0].device)
                # transformed_score = torch.zeros(B,N,K,H).to(points[0].device)
                # Correct
                # origin = torch.gather(index_prev,dim=-2,index=topk_idx)
                # N,k,H,K
                st1 = time.time()
                index_prev_mask = torch.stack([index_prev, index_mask], dim=0)
                origin_index_mask = index_prev_mask[:, topk_idx]

                # TODO inverse no batch
                # TODO the difference between the shape of score and idx
                # N,k,H
                kv_topk_score = topk_score[inverse_prev, :, :].permute(2, 0, 1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                                     k).reshape(H, -1,
                                                                                                                k * K)
                # TODO inverse no batch
                # H,M,K*k
                kv_topk_idx_mask = origin_index_mask[:, inverse_prev, :, :].transpose(-1, -2).reshape(2, N, -1,
                                                                                                      H).permute(0, 3,
                                                                                                                 1, 2)

                kv_topk_idx, kv_topk_mask = kv_topk_idx_mask[0], kv_topk_idx_mask[1].to(torch.bool)
                # kv_topk_idx = origin[inverse_prev,:,:].transpose(-1,-2).reshape(N,-1,H).permute(2,0,1)
                # kv_topk_mask = origin_mask[inverse_prev,:,:].transpose(-1,-2).reshape(N,-1,H).permute(2,0,1)
                # st = time.time()
                # index_query = torch.arange(M*H).reshape(H,M,1).repeat(1,1,k*K).to(kv_topk_idx.device)
                # print('arange',time.time() - st)
                # st = time.time()
                # index_query_1 = torch.arange(M*H).reshape(H,M,1).repeat(1,1,k*K).cuda()
                # print('arange_2',time.time() - st)
                index_softmax = torch.arange(M * K * H, device=device).reshape(H, M, K, 1).repeat(1, 1, 1, k).reshape(H,
                                                                                                                      M,
                                                                                                                      k * K)
                index_kv_base = torch.tensor([i * N for i in range(H)], device=device).long().reshape(H, 1, 1)
                kv_topk_idx_H = kv_topk_idx + index_kv_base

                topk_info = torch.stack([kv_topk_idx_H, kv_topk_idx, index_softmax], dim=0)
                topk_info = topk_info[:, kv_topk_mask]
                index_kv, kv_topk_idx, index_softmax = topk_info
                st = time.time()
                index_query = torch.div(index_softmax, K, rounding_mode='floor')
                # print('arange',time.time() - st)

                kv_topk_score = kv_topk_score[kv_topk_mask]

                # print('preprocess',time.time() - st1)

                final = True if i == len(points) - 1 else False

                A_fine, final_message, topk_score, topk_idx, topk_idx_prev = self.process_fine_level(
                    query, key, value, kv_topk_score, index_query, index_kv, index_softmax, kv_topk_idx, topk_prev,
                    topk, inverse_prev, index_prev, final
                )  # Quadtree attention
                # indices_new.append(topk_idx_prev)
                # final_message_fine = torch.zeros(B,N,self.dim).to(final_message.device)
            inverse_prev = inverse
            index_prev = index
            count_prev = count
            topk_prev = topk
            # if topk_idx is not None:
            #     topk_pos = torch.stack([topk_idx // w, topk_idx % w])  # convert to coordinate

        # indices_new.append(0)

        # final_message = 0
        # # should be modified
        # for i, m in enumerate(messages):
        #     if i == 0:
        #         final_message = m
        #     else:
        #         final_message = final_message.unsqueeze(2) + m
        #         final_message = rearrange(
        #             final_message, "b (H W) (t1 t2) h d -> b (H t1 W t2) h d", t1=2, t2=2, H=queries[-i].shape[2]
        #         )
        final_message = self.norm2(final_message)
        final_message = final_message + self.drop_path(self.mlp(final_message))

        # indices_new should be further researched, now deleted (indices_new just the list with 0 elements)
        return final_message, A_coarse, list(reversed(indices_new))


class QTAttB_test(nn.Module):
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

        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=0.0)
        self.dim = dim
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.in_dim = in_dim
        self.qkv = nn.Linear(in_dim, dim * 3, bias=qkv_bias)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm1 = norm_layer(in_dim)
        self.norm2 = norm_layer(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_points = num_points
        self.fc = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=in_dim, act_layer=act_layer,
                      drop=drop)
        self.latlayer = nn.Linear(dim, dim)

    def process_coarse_level(self, query, key, value, topk):
        # Correct
        M, H, C = query.shape
        N, H, C = key.shape

        # B,Number_Queries,Number_Keys,H
        # Correct
        QK = torch.einsum("lhd,shd->lsh", query, key)

        judge = torch.isnan(QK)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        softmax_temp = 1.0 / (C ** 0.5)  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)

        # plt.imshow(A.cpu().numpy()[1,:,:,0])
        # plt.colorbar()
        # plt.show()

        judge = torch.isnan(A)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        score = A.clone()

        A = self.attn_drop(A)
        message = torch.einsum("lsh,shd->lhd", A, value).reshape(N, H * C)  # .reshape(bs, h, w, self.nhead, cur_dim)
        message = self.proj(message)
        message = self.proj_drop(message)
        value = value.reshape(N, H * C)
        message = value.squeeze(1) + message

        judge = torch.isnan(message)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        return A, message, score

    # query, key, value, kv_topk_score, topk_prev, topk, inverse_prev, index_prev
    def process_fine_full_level(self, query, key, value, topk_score, topk_prev, topk, inverse_prev, index_prev):

        M, H, C = query.shape
        N, H, C = key.shape

        QK = torch.einsum("lhd,shd->lsh", query, key)

        judge = torch.isnan(QK)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        softmax_temp = 1.0 / (C ** 0.5)  # sqrt(D)

        # softmax_attn = scatter_softmax(QK * softmax_temp, inverse_prev, -2)

        # softmax_attn = softmax_attn * topk_score

        softmax_attn = torch.softmax(QK * softmax_temp, dim=-2)

        judge = torch.isnan(softmax_attn)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        softmax_attn = self.attn_drop(softmax_attn)
        message = torch.einsum("lsh,shd->lhd", softmax_attn, value).reshape(N,
                                                                            H * C)  # .reshape(bs, h, w, self.nhead, cur_dim)
        message = self.proj(message)
        message = self.proj_drop(message)
        value = value.reshape(N, H * C)
        message = value.squeeze(1) + message

        # print('message',message.shape)

        judge = torch.isnan(message)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        return softmax_attn, message, topk_score

    def forward(self, points, indices, inverses, slens, counts):
        """Multi-head quadtree attention
        Args:
            queries: Query pyramid [N, C, H, W]
            keys: Key pyramid [N, C, H, W]
            values: Value pyramid [N, C, H, W]
        Returns:
            message: (N, C, H, W)
        """

        topk = self.topks[0]
        indices_new = []
        messages = []
        for i, (point, index, inverse, slen, count) in enumerate(
                zip(reversed(points), reversed(indices), reversed(inverses), reversed(slens), reversed(counts))):
            # point = self.norm1(point)
            topk = self.topks[i]
            device = point.device
            if i == 0:
                N, C = point.shape
                # if N < topk * 4:
                #     topk = max(N // 4, 1)
                #     # print('coarse N < 32 with',point.shape)
                # qkv = self.qkv(point).reshape(N, 3, self.nhead, self.dim // self.nhead).permute(1, 0, 2, 3)
                # query, key, value = qkv[0], qkv[1], qkv[2]
                # # topk_idx B,Number_Queries,K,H
                # # Correct
                # st = time.time()
                # A_coarse, final_message, topk_score = self.process_coarse_level(
                #     query, key, value, topk
                # )  # Full attention for coarest level
                # # print('coarse_time',time.time() - st)
            else:
                N, C = point.shape
                # final_message = final_message[inverse_prev]
                # judge = torch.isnan(final_message)
                # judge_1 = torch.zeros_like(judge)
                # result = torch.all(torch.eq(judge_1,judge))
                # assert result == True
                qkv = self.qkv(point).reshape(N, 3, self.nhead, self.dim // self.nhead).permute(1, 0, 2, 3)
                query, key, value = qkv[0], qkv[1], qkv[2]
                H, M, C = query.shape
                H, N, C = key.shape
                judge = torch.isnan(qkv)
                judge_1 = torch.zeros_like(judge)
                result = torch.all(torch.eq(judge_1, judge))
                assert result == True
                # _,k = index_prev.shape

                # kv_topk_score = topk_score[inverse_prev,:]
                # kv_topk_score = kv_topk_score[:,inverse_prev,:]

                # final_message, xatt_weights_s = self.multihead_attn(query=point.unsqueeze(1),
                #                                    key=point.unsqueeze(1),
                #                                    value=point.unsqueeze(1))

                # final_message = final_message.squeeze(1)

                A, final_message, topk_idx = self.process_fine_full_level(
                    query, key, value, None, topk_prev, topk, inverse_prev, index_prev
                )  # Quadtree attention
                final_message = self.norm2(final_message)
                final_message = final_message + self.drop_path(self.mlp(final_message))
                # indices_new.append(topk_idx_prev)
            # final_message_fine = torch.zeros(B,N,self.dim).to(final_message.device)
            inverse_prev = inverse
            index_prev = index
            count_prev = count
            topk_prev = topk
            # final_message = self.norm2(final_message)
            # final_message = final_message + self.drop_path(self.mlp(final_message))
            # messages.append(final_message)
            # if topk_idx is not None:
            #     topk_pos = torch.stack([topk_idx // w, topk_idx % w])  # convert to coordinate

        # for i, (message, inverse) in enumerate(zip(messages,reversed(inverses))):
        #     if i == 0:
        #         final_message = self.latlayer(message)[inverse]
        #     elif i != len(messages) - 1:
        #         final_message = (final_message + self.latlayer(message))[inverse]
        #     else:
        #         final_message = final_message + self.latlayer(message)

        # indices_new.append(0)

        # final_message = 0
        # # should be modified
        # for i, m in enumerate(messages):
        #     if i == 0:
        #         final_message = m
        #     else:
        #         final_message = final_message.unsqueeze(2) + m
        #         final_message = rearrange(
        #             final_message, "b (H W) (t1 t2) h d -> b (H t1 W t2) h d", t1=2, t2=2, H=queries[-i].shape[2]
        #         )
        # message.append(final_message)

        # indices_new should be further researched, now deleted (indices_new just the list with 0 elements)
        return final_message, A, list(reversed(indices_new))


class QTAttB(nn.Module):
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
        # Correct
        B, M, H, C = query.shape
        B, N, H, C = key.shape

        # B,Number_Queries,Number_Keys,H
        # Correct
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        softmax_temp = 1.0 / (C ** 0.5)  # sqrt(D)

        if ONE_BATCH == False:
            mask_mat = key_padding_mask.unsqueeze(-2).repeat(1, M, 1)
            mask_mat = mask_mat.unsqueeze(-1).repeat(1, 1, 1, H)
            QK[mask_mat] = -1e9

        A = torch.softmax(softmax_temp * QK, dim=-2)

        A_mean = torch.mean(A, dim=-1)

        _, topk_idx = torch.topk(A_mean, dim=-1, k=topk, largest=True)

        # plt.imshow(A.cpu().numpy()[1,:,:,0])
        # plt.colorbar()
        # plt.show()

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
        # Correct
        B, M, H, C = query.shape
        B, N, H, C = key.shape

        # B,Number_Queries,Number_Keys,H
        # Correct
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        softmax_temp = 1.0 / (C ** 0.5)  # sqrt(D)
        mask_mat = key_padding_mask.unsqueeze(-2).repeat(1, M, 1)
        mask_mat = mask_mat.unsqueeze(-1).repeat(1, 1, 1, H)

        QK[mask_mat] = -1e9
        A = torch.softmax(softmax_temp * QK, dim=-2)

        # plt.imshow(A.cpu().numpy()[1,:,:,0])
        # plt.colorbar()
        # plt.show()

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
        # max number of keys
        n_max = index_query_counts.max()
        # offset
        index_query_counts = index_query_counts.cumsum(dim=-1)  # [N]

        # why N+1
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
        # message = value_aggregation_op(A, value.contiguous(), kv_topk_idx)

        # should be modified
        attn_matrix = softmax_attn_flat.reshape(H * M, -1)

        topk_score, topk_idx = torch.topk(attn_matrix, dim=-1, k=topk, largest=True)

        # H,M,K -> H*M,K
        topk_idx = (index_query_offsets[:-1].unsqueeze(-1) + topk_idx).reshape(-1, 1)

        topk_idx = torch.gather(kv_topk_idx.unsqueeze(-1), index=topk_idx, dim=0).reshape(H, M, topk).permute(1, 2, 0)
        topk_score = topk_score.reshape(H, M, topk).permute(1, 2, 0)
        # topk_idx_prev = torch.gather(index_kv, index=topk_idx_prev, dim=-2)

        # self_idx = torch.arange(topk_idx.shape[1]).reshape(1,-1,1,1).repeat(B,1,1,H)
        # topk_idx = torch.cat([topk_idx,self_idx],dim=-2)

        # topk_idx: B,h//2,w//2,2,2,K_New,H
        # topk_idx = rearrange(topk_idx, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2)  # reshape back
        # topk_score = rearrange(
        #     topk_score, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2
        # )  # reshape back
        return softmax_attn_flat, message, topk_score, topk_idx, None

    def process_fine_level(self, query, key, value, index_query, index_kv, topk, final):
        topk_idx = None
        device = 'cuda'

        M, H, C = query.shape
        N, H, C = key.shape

        index_query_counts = index_query.bincount()
        # max number of keys
        n_max = index_query_counts.max()
        # offset
        index_query_counts = index_query_counts.cumsum(dim=-1)  # [N]

        # why N+1
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
        # message = value_aggregation_op(A, value.contiguous(), kv_topk_idx)

        return softmax_attn_flat, message, topk_idx

    def forward(self, points, indices, inverses, slens, counts):
        """Multi-head quadtree attention
        Args:
            queries: Query pyramid [N, C, H, W]
            keys: Key pyramid [N, C, H, W]
            values: Value pyramid [N, C, H, W]
        Returns:
            message: (N, C, H, W)
        """

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
                    # padding_mask: masked locations are True
                    point, padding_mask, _ = pad_sequence(separate, require_padding_mask=True, batch_first=True)
                B, N, C = point.shape
                if 2 * topk >= N:
                    topk = int(N//2)
                qkv = self.qkv(point).reshape(B, N, 3, self.nhead, self.dim // self.nhead).permute(2, 0, 1, 3, 4)
                query, key, value = qkv[0], qkv[1], qkv[2]
                # topk_idx B,Number_Queries,K,H
                # Correct
                st = time.time()
                A_coarse, final_message, topk_idx = self.process_coarse_level(
                    query, key, value, topk, padding_mask, padding_mask
                )  # Full attention for coarest level
                # print('coarse_time',time.time() - st)

                final_message = self.norm2(final_message)
                # final_message = final_message + self.drop_path(self.mlp_coarse(final_message))
            else:
                N, C = point.shape
                message_prev = final_message[inverse_prev]

                qkv = self.qkv(point + self.fc(message_prev)).reshape(N, 3, self.nhead,
                                                                       self.dim // self.nhead).permute(1, 0, 2, 3)
                # qkv = self.qkv_fine(point).reshape(N, 3, self.nhead, self.dim // self.nhead).permute(1, 0, 2, 3)
                query, key, value = qkv[0], qkv[1], qkv[2]
                M, H, C = query.shape
                N, H, C = key.shape

                _, k = index_prev.shape
                N_prev, K = topk_idx.shape

                # mask = torch.arange(k,device=device).unsqueeze(0).cuda() < count_prev.unsqueeze(-1) #[n, k]
                # mask_mat = (mask.unsqueeze(-1) & mask.unsqueeze(-2)) #[n, k, k]
                # index_0 = index_prev.unsqueeze(-1).expand(-1, -1, k)[mask_mat] #[M, ]
                # index_1 = index_prev.unsqueeze(1).expand(-1, k, -1)[mask_mat] #[M, ]
                # index_0, indices = torch.sort(index_0) #[M,]
                # # index_1 -> key
                # index_1 = index_1[indices] #[M,]

                # mask = torch.arange(k,device=device).unsqueeze(0).cuda() < count_prev.unsqueeze(-1) #[n, k]
                # index_mask = torch.stack([index_prev,mask],dim=0)
                # index_mask = index_mask[:,inverse_prev]
                # index_1, mask = index_mask[0], index_mask[1].to(torch.bool)
                # # index_1 = index_prev[inverse_prev,:]
                # # mask = mask[inverse_prev,:]
                # index_0 = torch.arange(M,device=device).reshape(M,1).repeat(1,k)
                # index_01 = torch.stack([index_0,index_1],dim=0)
                # index_01 = index_01[:,mask]
                # index_0, index_1 = index_01[0], index_01[1]

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
                )  # Quadtree attention
                final_message = self.norm2(final_message)
                final_message = final_message + self.drop_path(self.mlp_fine(final_message))

                final_message = final_message + message_prev

                # indices_new.append(topk_idx_prev)
            # final_message_fine = torch.zeros(B,N,self.dim).to(final_message.device)
            inverse_prev = inverse
            index_prev = index
            count_prev = count
            topk_prev = topk
            messages.append(final_message)
            # if topk_idx is not None:
            #     topk_pos = torch.stack([topk_idx // w, topk_idx % w])  # convert to coordinate

        # for i, (message, inverse) in enumerate(zip(messages,reversed(inverses))):
        #     if i == 0:
        #         final_message = self.latlayer(message)[inverse]
        #     elif i != len(messages) - 1:
        #         final_message = (final_message + self.latlayer(message))[inverse]
        #     else:
        #         final_message = final_message + self.latlayer(message)

        # indices_new.append(0)

        # final_message = 0
        # # should be modified
        # for i, m in enumerate(messages):
        #     if i == 0:
        #         final_message = m
        #     else:
        #         final_message = final_message.unsqueeze(2) + m
        #         final_message = rearrange(
        #             final_message, "b (H W) (t1 t2) h d -> b (H t1 W t2) h d", t1=2, t2=2, H=queries[-i].shape[2]
        #         )
        # message.append(final_message)

        # indices_new should be further researched, now deleted (indices_new just the list with 0 elements)
        return final_message, A_coarse, list(reversed(indices_new)), messages[:-1]


class QTAttB_cross(nn.Module):
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
        # Correct
        B, M, H, C = query.shape
        B, N, H, C = key.shape

        # B,Number_Queries,Number_Keys,H
        # Correct
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        softmax_temp = 1.0 / (C ** 0.5)  # sqrt(D)

        # mask_mat = key_padding_mask.unsqueeze(-2).repeat(1,M,1)
        # mask_mat = mask_mat.unsqueeze(-1).repeat(1,1,1,H)
        # QK[mask_mat] = -1e9

        A = torch.softmax(softmax_temp * QK, dim=-2)

        A_mean = torch.mean(A, dim=-1)

        _, topk_idx = torch.topk(A_mean, dim=-1, k=topk, largest=True)

        # plt.imshow(A.cpu().numpy()[1,:,:,0])
        # plt.colorbar()
        # plt.show()

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
        # Correct
        B, M, H, C = query.shape
        B, N, H, C = key.shape

        # B,Number_Queries,Number_Keys,H
        # Correct
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

        # plt.imshow(A.cpu().numpy()[1,:,:,0])
        # plt.colorbar()
        # plt.show()

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
        # max number of keys
        n_max = index_query_counts.max()
        # offset
        index_query_counts = index_query_counts.cumsum(dim=-1)  # [N]

        # why N+1
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
        # message = value_aggregation_op(A, value.contiguous(), kv_topk_idx)

        judge = torch.isnan(message)
        judge_1 = torch.zeros_like(judge)
        result = torch.all(torch.eq(judge_1, judge))
        assert result == True

        # should be modified
        attn_matrix = softmax_attn_flat.reshape(H * M, -1)

        topk_score, topk_idx = torch.topk(attn_matrix, dim=-1, k=topk, largest=True)

        # H,M,K -> H*M,K
        topk_idx = (index_query_offsets[:-1].unsqueeze(-1) + topk_idx).reshape(-1, 1)

        topk_idx = torch.gather(kv_topk_idx.unsqueeze(-1), index=topk_idx, dim=0).reshape(H, M, topk).permute(1, 2, 0)
        topk_score = topk_score.reshape(H, M, topk).permute(1, 2, 0)
        # topk_idx_prev = torch.gather(index_kv, index=topk_idx_prev, dim=-2)

        # self_idx = torch.arange(topk_idx.shape[1]).reshape(1,-1,1,1).repeat(B,1,1,H)
        # topk_idx = torch.cat([topk_idx,self_idx],dim=-2)

        # topk_idx: B,h//2,w//2,2,2,K_New,H
        # topk_idx = rearrange(topk_idx, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2)  # reshape back
        # topk_score = rearrange(
        #     topk_score, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2
        # )  # reshape back
        return softmax_attn_flat, message, topk_score, topk_idx, None

    def process_fine_level(self, query, key, value, q_value, index_query, index_kv, index_query_counts, topk, final):

        M, H, C = query.shape
        N, H, C = key.shape

        topk_idx = None

        device = query.device

        # index_query_counts = index_query.bincount()
        # max number of keys
        n_max = index_query_counts.max()
        # offset
        index_query_counts = index_query_counts.cumsum(dim=-1)  # [N]

        # why N+1
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
        # message = value_aggregation_op(A, value.contiguous(), kv_topk_idx)

        return softmax_attn_flat, message, topk_idx

    def coarse_forward(self, point_q, point_kv, topk):
        B, M, C = point_q.shape
        B, N, C = point_kv.shape
        kv = self.kv(point_kv).reshape(B, N, 2, self.nhead, self.dim // self.nhead).permute(2, 0, 1, 3, 4)
        key, value = kv[0], kv[1]
        q = self.q(point_q).reshape(B, M, 2, self.nhead, self.dim // self.nhead).permute(2, 0, 1, 3, 4)
        query, q_value = q[0], q[1]
        # topk_idx B,Number_Queries,K,H
        # Correct
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
        )  # Quadtree attention
        final_message = self.norm2(final_message)
        final_message = final_message + self.drop_path(self.mlp_fine(final_message))
        return A_fine, final_message, topk_idx

    def forward(self, points_src, points_tgt, indices_src, indices_tgt, inverses_src, inverses_tgt, counts_src,
                counts_tgt):
        """Multi-head quadtree attention
        Args:
            queries: Query pyramid [N, C, H, W]
            keys: Key pyramid [N, C, H, W]
            values: Value pyramid [N, C, H, W]
        Returns:
            message: (N, C, H, W)
        """

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
                # indices_new.append(topk_idx_prev)
            # final_message_fine = torch.zeros(B,N,self.dim).to(final_message.device)
            inverse_prev_src = inverse_src
            index_prev_src = index_src
            count_prev_src = count_src
            inverse_prev_tgt = inverse_tgt
            index_prev_tgt = index_tgt
            count_prev_tgt = count_tgt
            src_messages.append(src_message)
            tgt_messages.append(tgt_message)
            # if topk_idx is not None:
            #     topk_pos = torch.stack([topk_idx // w, topk_idx % w])  # convert to coordinate

        # for i, (message, inverse) in enumerate(zip(messages,reversed(inverses))):
        #     if i == 0:
        #         final_message = self.latlayer(message)[inverse]
        #     elif i != len(messages) - 1:
        #         final_message = (final_message + self.latlayer(message))[inverse]
        #     else:
        #         final_message = final_message + self.latlayer(message)

        # indices_new.append(0)

        # final_message = 0
        # # should be modified
        # for i, m in enumerate(messages):
        #     if i == 0:
        #         final_message = m
        #     else:
        #         final_message = final_message.unsqueeze(2) + m
        #         final_message = rearrange(
        #             final_message, "b (H W) (t1 t2) h d -> b (H t1 W t2) h d", t1=2, t2=2, H=queries[-i].shape[2]
        #         )
        # message.append(final_message)

        # indices_new should be further researched, now deleted (indices_new just the list with 0 elements)
        return src_message, tgt_message, src_A_coarse, tgt_A_coarse, src_messages[:-1], tgt_messages[:-1]
