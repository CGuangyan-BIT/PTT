"""Modified from DETR's transformer.py

- Cross encoder layer is similar to the decoder layers in Transformer, but
  updates both source and target features
- Added argument to control whether value has position embedding or not for
  TransformerEncoderLayer and TransformerDecoderLayer
- Decoder layer now keeps track of attention weights
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import sys

import torch_scatter

from models.transformer.PointTreeAttention import PTA, PTA_cross, Mlp, ONE_BATCH

import logging
import open3d as o3d
import numpy as np
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
from models.transformer.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned


class Tree_Transformer(nn.Module):

    def __init__(self, cross_encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src, tgt, Tree_info,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_pcd: Optional[Tensor] = None,
                tgt_pcd: Optional[Tensor] = None):

        src_intermediate, tgt_intermediate = [], []

        src_messages, tgt_messages = None, None

        for i, layer in enumerate(self.layers):
            src, tgt, src_messages, tgt_messages = layer(src=src, tgt=tgt, Tree_info=Tree_info, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos, tgt_pos=tgt_pos,
                             src_messages=src_messages,
                             tgt_messages=tgt_messages,
                             layer_index=i)
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(tgt)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src.unsqueeze(0), tgt.unsqueeze(0)

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)


class Tree_TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod',
                 knn_k = 8
                 ):
        super().__init__()

        self.knn_k = knn_k

        self.logger = logging.getLogger(self.__class__.__name__)


        if attention_type == 'dot_prod':
            self.self_attn = PTA(nhead, in_dim=d_model, dim=d_model, topks=[8,8,8], knn_k=self.knn_k)

            self.cross_attn = PTA_cross(nhead, in_dim=d_model, dim=d_model, topks=[8,8,8], knn_k=self.knn_k)
        else:
            raise NotImplementedError


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
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

        self.linear3_self = nn.Linear(d_model+3,dim_feedforward)
        self.linear3_cross = nn.Linear(d_model + 3, dim_feedforward)
        self.linear4_self = nn.Linear(dim_feedforward, d_model)
        self.linear4_cross = nn.Linear(dim_feedforward, d_model)
        self.mlp = Mlp(in_features=d_model, hidden_features=int(dim_feedforward), out_features=d_model, act_layer=nn.GELU, drop=0)
        self.norm4_self = nn.LayerNorm(d_model)
        self.norm4_cross = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.activation1 = _get_activation_fn(activation)

    def feature_scatter(self,feature,inverse_list,pe_list=None):
        feature_list = []
        if pe_list != None:
            feature_list.append(feature+pe_list[0])
        else:
            feature_list.append(feature)
        for i, inverse in enumerate(inverse_list):
            feature_new = torch_scatter.scatter_sum(feature, inverse, dim=0)
            if pe_list != None:
                feature_list.append(feature_new+pe_list[i+1])
            else:
                feature_list.append(feature_new)
            feature = feature_new
        return feature_list

    def feature_scatter_conv_self(self,feature,inverse_list,pcd_list,pe_list=None,messages=None):
        feature_list = []
        if messages != None:
            len_m = len(messages)
        if pe_list != None:
            feature_list.append(feature+pe_list[0])
        else:
            feature_list.append(feature)
        for i, inverse in enumerate(inverse_list):
            pcd_fine = pcd_list[i]
            pcd_coarse = pcd_list[i+1][inverse]
            feature = torch.cat([feature,pcd_fine-pcd_coarse],dim=-1)
            feature = self.linear4_self(self.dropout4(self.activation1(self.linear3_self(feature))))
            feature = self.norm4_self(feature)
            feature_new = torch_scatter.scatter_mean(feature, inverse, dim=0)
            if messages != None:
                feature_new = messages[len_m-i-1] + feature_new
            if pe_list != None:
                feature_list.append(feature_new+pe_list[i+1])
            else:
                feature_list.append(feature_new)
            feature = feature_new
        return feature_list

    def feature_scatter_conv_cross(self, feature, inverse_list, pcd_list, pe_list=None, messages=None):
        feature_list = []
        if messages != None:
            len_m = len(messages)
        if pe_list != None:
            feature_list.append(feature+pe_list[0])
        else:
            feature_list.append(feature)
        for i, inverse in enumerate(inverse_list):
            pcd_fine = pcd_list[i]
            pcd_coarse = pcd_list[i+1][inverse]
            feature = torch.cat([feature, pcd_fine-pcd_coarse], dim=-1)
            feature = self.linear4_cross(self.dropout4(self.activation1(self.linear3_cross(feature))))
            feature = self.norm4_cross(feature)
            feature_new = torch_scatter.scatter_mean(feature, inverse, dim=0)
            if messages != None:
                feature_new = messages[len_m-i-1] + feature_new
            if pe_list != None:
                feature_list.append(feature_new+pe_list[i+1])
            else:
                feature_list.append(feature_new)
            feature = feature_new
        return feature_list

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src, tgt, Tree_info,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,
                    src_messages: Optional[Tensor] = None,
                    tgt_messages: Optional[Tensor] = None,
                    layer_index = None):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        suffix = str(layer_index % 1 + 1) 

        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)

        src_list = self.feature_scatter_conv_self(src2_w_pos,Tree_info[f'inverse_src'][1:],Tree_info[f'pcd_src'],None,src_messages)

        src2, satt_weights_s, _, src_messages = self.self_attn(src_list,Tree_info[f'index_src'],Tree_info[f'inverse_src'],Tree_info[f'slen_src'],Tree_info[f'counts_src'])

        src = src + self.dropout1(src2)
        


        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        tgt_list = self.feature_scatter_conv_self(tgt2_w_pos, Tree_info[f'inverse_tgt'][1:], Tree_info[f'pcd_tgt'], None, tgt_messages)

        tgt, satt_weights_t, _, tgt_messages = self.self_attn(tgt_list, Tree_info[f'index_tgt'], Tree_info[f'inverse_tgt'], Tree_info[f'slen_tgt'], Tree_info[f'counts_tgt'])
        tgt = tgt + self.dropout1(tgt2)

        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        src_list = self.feature_scatter_conv_cross(src_w_pos, Tree_info[f'inverse_src'][1:], Tree_info[f'pcd_src'], None, src_messages)
        tgt_list = self.feature_scatter_conv_cross(tgt_w_pos, Tree_info[f'inverse_tgt'][1:], Tree_info[f'pcd_tgt'], None, tgt_messages)


        src3, tgt3, xatt_weights_s, xatt_weights_t, src_messages, tgt_messages = \
            self.cross_attn(src_list, tgt_list,
                            Tree_info[f'index_src'], Tree_info[f'index_tgt'],
                            Tree_info[f'inverse_src'], Tree_info[f'inverse_tgt'],
                            Tree_info[f'counts_src'], Tree_info[f'counts_tgt'])

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)


        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)


        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt, src_messages, tgt_messages

    def forward(self, src, tgt, Tree_info,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_messages: Optional[Tensor] = None,
                tgt_messages: Optional[Tensor] = None,
                layer_index = None):

        if self.normalize_before:
            return self.forward_pre(src=src, tgt=tgt, Tree_info=Tree_info, src_mask=src_mask, tgt_mask=tgt_mask,
                                    src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, 
                                    src_pos=src_pos, tgt_pos=tgt_pos,
                                    src_messages=src_messages, tgt_messages=tgt_messages, layer_index=layer_index)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)

class TransformerCrossDecoder(nn.Module):

    def __init__(self, cross_encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src, tgt,
                src_mask_token: Optional[Tensor] = None,
                tgt_mask_token: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_mask_pos: Optional[Tensor] = None,
                tgt_mask_pos: Optional[Tensor] = None,
                ):

        src_intermediate, tgt_intermediate = [], []

        for layer in self.layers:
            src, tgt, src_mask_token, tgt_mask_token = \
                layer(src, tgt, src_mask_token=src_mask_token, tgt_mask_token=tgt_mask_token,
                src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                src_pos=src_pos, tgt_pos=tgt_pos, src_mask_pos=src_mask_pos, tgt_mask_pos=tgt_mask_pos)

        if self.norm is not None:
            src_mask_token = self.norm(src_mask_token)
            tgt_mask_token = self.norm(tgt_mask_token)

        return src_mask_token.unsqueeze(0), tgt_mask_token.unsqueeze(0)

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)


class TransformerCrossDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()


        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError


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
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'


        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)


        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))


        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)


        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask_token: Optional[Tensor] = None,
                    tgt_mask_token: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,
                    src_mask_pos: Optional[Tensor] = None,
                    tgt_mask_pos: Optional[Tensor] = None,
                    ):


        src_len = src.shape[0]
        tgt_len = tgt.shape[0]
        src = torch.cat([src, src_mask_token], dim=0)
        tgt = torch.cat([tgt, tgt_mask_token], dim=0)
        src_pos = torch.cat([src_pos, src_mask_pos], dim=0)
        tgt_pos = torch.cat([tgt_pos, tgt_mask_pos], dim=0)

        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        q = k = src2_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                                              value=src2_w_pos if self.sa_val_has_pos_emb else src2,
                                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)
        q = k = tgt2_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)


        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)


        return src[:src_len], tgt[:tgt_len], src[src_len:], tgt[tgt_len:]

    def forward(self, src, tgt,
                src_mask_token: Optional[Tensor] = None,
                tgt_mask_token: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_mask_pos: Optional[Tensor] = None,
                tgt_mask_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask_token, tgt_mask_token,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos,
                                    src_mask_pos, tgt_mask_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

