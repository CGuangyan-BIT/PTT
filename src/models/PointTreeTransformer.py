"""REGTR-2 network architecture
"""
import math

import torch
import torch.nn as nn

from models.backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps, batch_knn_kpconv_gpu
from models.generic_reg_model import GenericRegModel
from models.losses.corr_loss import CorrCriterion
from models.losses.feature_loss import InfoNCELossFull, CircleLossFull
from models.transformer.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned
from models.transformer.transformers import \
    Tree_TransformerLayer, Tree_Transformer, TransformerCrossDecoderLayer, TransformerCrossDecoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv, se3_transform
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
from utils.viz import visualize_registration
from pytorch3d.ops import knn_points
import torch_points_kernels as tp
from torch_geometric.nn import voxel_grid
import torch_scatter
import open3d as o3d
import numpy as np
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import matplotlib.pyplot as plt
_TIMEIT = False
ONE_BATCH = True

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

class Group(nn.Module):
    def __init__(self, num_group, group_size, mask_ratio):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.mask_ratio = mask_ratio

    def forward(self, xyz_list, pose):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        xyz, _, _ = pad_sequence(xyz_list, require_padding_mask=False)
        xyz = xyz.transpose(0, 1).contiguous()
        B, N, _ = xyz.shape
        xyz = se3_transform(pose, xyz).contiguous()
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group) # B G 3
        _, idx_n = self.knn(xyz, center) # B G M
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx_n + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        return neighborhood, center

class PTT(GenericRegModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.group_size = 16

        self.group = Group(64, self.group_size, [0.1, 0.4])
        self.preprocessor = PreprocessorGPU(cfg)
        self.kpf_encoder = KPFEncoder(cfg, cfg.d_embed)
        self.feat_proj = nn.Linear(self.kpf_encoder.encoder_skip_dims[-1], cfg.d_embed, bias=True)
        if cfg.get('pos_emb_type', 'sine') == 'sine':
            self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.d_embed,
                                                         scale=cfg.get('pos_emb_scaling', 1.0))
        elif cfg['pos_emb_type'] == 'learned':
            self.pos_embed = PositionEmbeddingLearned(3, cfg.d_embed)
        else:
            raise NotImplementedError
        self.knn_k = 20
        self.level = cfg.layers - 1
        self.fine_voxel_size = cfg.fine_voxel_size # 0.3 for 3dmatch
        encoder_layer = Tree_TransformerLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=cfg.sa_val_has_pos_emb,
            ca_val_has_pos_emb=cfg.ca_val_has_pos_emb,
            attention_type=cfg.attention_type,
            knn_k=self.knn_k
        )
        self.growing = cfg.growing

        encoder_norm = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        self.transformer_encoder = Tree_Transformer(
            encoder_layer, cfg.num_encoder_layers, encoder_norm,
            return_intermediate=True)

        decoder_layer = TransformerCrossDecoderLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=cfg.sa_val_has_pos_emb,
            ca_val_has_pos_emb=cfg.ca_val_has_pos_emb,
            attention_type=cfg.attention_type,
        )

        decoder_norm = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        self.transformer_decoder = TransformerCrossDecoder(
            decoder_layer, cfg.num_decoder_layers, decoder_norm,
            return_intermediate=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_embed))
        if cfg.get('direct_regress_coor', False):
            self.correspondence_decoder = CorrespondenceRegressor(cfg.d_embed)
            self.mask_decoder = MaskRegressor(cfg.d_embed, self.group_size)
        else:
            self.correspondence_decoder = CorrespondenceDecoder(cfg.d_embed,
                                                                cfg.corr_decoder_has_pos_emb,
                                                                self.pos_embed)
        self.overlap_criterion = nn.BCEWithLogitsLoss()
        if self.cfg.feature_loss_type == 'infonce':
            self.feature_criterion = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
        elif self.cfg.feature_loss_type == 'circle':
            self.feature_criterion = CircleLossFull(dist_type='euclidean', r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = self.feature_criterion
        else:
            raise NotImplementedError

        self.corr_criterion = CorrCriterion(metric='mae')

        self.chamfer_loss = ChamferDistanceL2().cuda()

        self.weight_dict = {}
        for k in ['overlap', 'feature', 'corr', 'mask_corr']:
            for i in cfg.get(f'{k}_loss_on', [cfg.num_encoder_layers - 1]):
                self.weight_dict[f'{k}_{i}'] = cfg.get(f'wt_{k}')
        self.weight_dict['feature_un'] = cfg.wt_feature_un

        self.logger.info('Loss weighting: {}'.format(self.weight_dict))
        self.logger.info(
            f'Config: d_embed:{cfg.d_embed}, nheads:{cfg.nhead}, pre_norm:{cfg.pre_norm}, '
            f'use_pos_emb:{cfg.transformer_encoder_has_pos_emb}, '
            f'sa_val_has_pos_emb:{cfg.sa_val_has_pos_emb}, '
            f'ca_val_has_pos_emb:{cfg.ca_val_has_pos_emb}'
        )

    def grid_sample(self,pos, batch, size, start, return_p2v=True):
        cluster = voxel_grid(pos, batch, size, start=start) #[N, ]

        un, inverse, count = torch.unique(batch, sorted=True, return_inverse=True, return_counts=True)

        if return_p2v == False:
            unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
            return cluster

        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        n = unique.shape[0]
        k = counts.max().item()
        p2v_map = cluster.new_zeros(n, k) #[n, k]
        mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1) #[n, k]
        p2v_map[mask] = torch.argsort(cluster)
        return cluster, p2v_map, counts

    def collect_points(self,points,slens_c,bias=False):
        index_list = []
        inverse_list = []
        pe_list = []
        index_list.append(0)
        inverse_list.append(0)
        slens_list = []
        slens_list.append(slens_c)
        points_list = []
        points_list.append(points)
        pe_list.append(self.pos_embed(points))
        counts_list = []
        counts_list.append(0)

        for i in range(self.level):
            if ONE_BATCH == True:
                batch = torch.ones(points.shape[0]).long().cuda()
            else:
                batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(slens_c)], 0).long().cuda()

            window_size = torch.tensor([self.fine_voxel_size * (self.growing**i)]*3).type_as(points).to(points.device)

            if bias == True:
                v2p_map, p2v_map, counts = self.grid_sample(points + torch.tensor([self.fine_voxel_size/2]*3,device='cuda').reshape(1,-1), batch, window_size, start=None)
            else:
                v2p_map, p2v_map, counts = self.grid_sample(points, batch, window_size, start=None)
            
            points_new = torch_scatter.scatter_mean(points, v2p_map, dim=0)

            if ONE_BATCH == False:
                slenc_new = torch.tensor([max(v2p_map[:sum(slens_c[:i + 1])]) + 1 for i in range(len(slens_c))]).long()
                slenc_new[1:] = slenc_new[1:] - slenc_new[:-1]
                slens_c = slenc_new
                slens_list.append(slenc_new)
            else:
                slens_list.append(0)

            points = points_new

            index_list.append(p2v_map)
            inverse_list.append(v2p_map)
            points_list.append(points)
            counts_list.append(counts)
            pe_list.append(self.pos_embed(points))
        
        return index_list, inverse_list, points_list, slens_list, counts_list, pe_list

    def forward(self, batch, is_train=True):
        B = len(batch['src_xyz'])
        outputs = {}
        Tree_info = {}

        if _TIMEIT:
            t_start_all_cuda, t_end_all_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pp_cuda, t_end_pp_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_all_cuda.record()
            t_start_pp_cuda.record()
        kpconv_meta = self.preprocessor(batch['src_xyz'] + batch['tgt_xyz'])
        batch['kpconv_meta'] = kpconv_meta
        slens = [s.tolist() for s in kpconv_meta['stack_lengths']]
        slens_c = slens[-1]
        src_slens_c, tgt_slens_c = slens_c[:B], slens_c[B:]
        feats0 = torch.ones_like(kpconv_meta['points'][0][:, 0:1])

        if _TIMEIT:
            t_end_pp_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pp_cuda = t_start_pp_cuda.elapsed_time(t_end_pp_cuda) / 1000
            t_start_enc_cuda, t_end_enc_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_enc_cuda.record()
        feats_un, skip_x = self.kpf_encoder(feats0, kpconv_meta)
        if _TIMEIT:
            t_end_enc_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_enc_cuda = t_start_enc_cuda.elapsed_time(t_end_enc_cuda) / 1000
            t_start_att_cuda, t_end_att_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_att_cuda.record()

        src_xyz_c, tgt_xyz_c = split_src_tgt(kpconv_meta['points'][-1], slens_c)
        pe = self.pos_embed(kpconv_meta['points'][-1])
        both_feats_un = self.feat_proj(feats_un)

        if is_train:
            pose = torch.tensor(batch['pose'], device='cuda')
            inv_pose = se3_inv(pose)
            src_neighbor, src_center, = self.group(src_xyz_c, pose)
            tgt_neighbor, tgt_center, = self.group(tgt_xyz_c, inv_pose)
        src_pe, tgt_pe = split_src_tgt(pe, slens_c)
        src_feats_un, tgt_feats_un = split_src_tgt(both_feats_un, slens_c)
        src_pe_padded, _, _ = pad_sequence(src_pe)
        tgt_pe_padded, _, _ = pad_sequence(tgt_pe)
        
        src_feats_padded, src_key_padding_mask, _ = pad_sequence(src_feats_un,
                                                                 require_padding_mask=True)
        tgt_feats_padded, tgt_key_padding_mask, _ = pad_sequence(tgt_feats_un,
                                                                 require_padding_mask=True)

        B = len(slens_c) // 2

        src_xyz = torch.cat(src_xyz_c,dim=0)
        tgt_xyz = torch.cat(tgt_xyz_c,dim=0)


        with torch.no_grad():
            Tree_info['index_src'], Tree_info['inverse_src'], Tree_info['pcd_src'], Tree_info['slen_src'], Tree_info['counts_src'], Tree_info['pe_src'] = self.collect_points(src_xyz, slens_c[:B])
            Tree_info['index_tgt'], Tree_info['inverse_tgt'], Tree_info['pcd_tgt'], Tree_info['slen_tgt'], Tree_info['counts_tgt'], Tree_info['pe_tgt'] = self.collect_points(tgt_xyz, slens_c[B:])

        src_feats = torch.cat(src_feats_un,dim=0)
        tgt_feats = torch.cat(tgt_feats_un,dim=0)

        src_pe = torch.cat(src_pe,dim=0)
        tgt_pe = torch.cat(tgt_pe,dim=0)

        src_feats_cond, tgt_feats_cond = self.transformer_encoder(
            src=src_feats, tgt=tgt_feats, Tree_info=Tree_info,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_pos=src_pe if self.cfg.transformer_encoder_has_pos_emb else None,
            tgt_pos=tgt_pe if self.cfg.transformer_encoder_has_pos_emb else None,
        )

        separate = torch.split(src_feats_cond.transpose(0,1), list(slens_c[:B]), dim=0)
        src_feats_cond, _, _ = pad_sequence(separate, require_padding_mask=False, batch_first=False)

        separate = torch.split(tgt_feats_cond.transpose(0,1), list(slens_c[B:]), dim=0)
        tgt_feats_cond, _, _ = pad_sequence(separate, require_padding_mask=False, batch_first=False)

        src_feats_cond, tgt_feats_cond = src_feats_cond.permute(2,0,1,3), tgt_feats_cond.permute(2,0,1,3)   


        src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list = \
            self.correspondence_decoder(src_feats_cond, tgt_feats_cond, src_xyz_c, tgt_xyz_c)

        if is_train:
            src_num_mask = tgt_center.shape[1]
            tgt_num_mask = src_center.shape[1]
            src_mask_tokens = self.mask_token.repeat(src_num_mask, B, 1)
            tgt_mask_tokens = self.mask_token.repeat(tgt_num_mask, B, 1)

            src_mask_pos = self.pos_embed(tgt_center).transpose(0, 1)
            tgt_mask_pos = self.pos_embed(src_center).transpose(0, 1)

            b, g_src, _, _ = src_neighbor.shape
            b, g_tgt, _, _ = tgt_neighbor.shape

            src_mask_xyz = tgt_neighbor.reshape(b*g_tgt, -1, 3)
            tgt_mask_xyz = src_neighbor.reshape(b*g_src, -1, 3)

            src_masked_mask = torch.zeros(B, src_num_mask, device='cuda').to(torch.bool)
            tgt_masked_mask = torch.zeros(B, tgt_num_mask, device='cuda').to(torch.bool)

            src_key_padding_mask = torch.cat([src_key_padding_mask, src_masked_mask], dim=1)
            tgt_key_padding_mask = torch.cat([tgt_key_padding_mask, tgt_masked_mask], dim=1)

            src_mask_feats, tgt_mask_feats = self.transformer_decoder(
                src_feats_cond[-1], tgt_feats_cond[-1],
                src_mask_token=src_mask_tokens,
                tgt_mask_token=tgt_mask_tokens,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                src_pos=src_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
                tgt_pos=tgt_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
                src_mask_pos=src_mask_pos,
                tgt_mask_pos=tgt_mask_pos,
            )

            src_mask_corr, tgt_mask_corr = \
                self.mask_decoder(src_mask_feats, tgt_mask_feats, src_mask_xyz, tgt_mask_xyz)
        else:
            src_mask_xyz = None
            tgt_mask_xyz = None
            src_mask_corr = None
            tgt_mask_corr = None
            src_neighbor = None
            tgt_neighbor = None

        src_feats_list = unpad_sequences(src_feats_cond, src_slens_c)
        tgt_feats_list = unpad_sequences(tgt_feats_cond, tgt_slens_c)
        num_pred = src_feats_cond.shape[0]
        if _TIMEIT:
            t_end_att_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_att_cuda = t_start_att_cuda.elapsed_time(t_end_att_cuda) / 1000
            t_start_pose_cuda, t_end_pose_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pose_cuda.record()
        corr_all, overlap_prob = [], []
        for b in range(B):
            corr_all.append(torch.cat([
                torch.cat([src_xyz_c[b].expand(num_pred, -1, -1), src_corr_list[b]], dim=2),
                torch.cat([tgt_corr_list[b], tgt_xyz_c[b].expand(num_pred, -1, -1)], dim=2)
            ], dim=1))
            overlap_prob.append(torch.cat([
                torch.sigmoid(src_overlap_list[b][:, :, 0]),
                torch.sigmoid(tgt_overlap_list[b][:, :, 0]),
            ], dim=1))

        pred_pose_weighted = torch.stack([
            compute_rigid_transform(corr_all[b][..., :3], corr_all[b][..., 3:],
                                    overlap_prob[b])
            for b in range(B)], dim=1)
        if _TIMEIT:
            t_end_pose_cuda.record()
            t_end_all_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pose_cuda = t_start_pose_cuda.elapsed_time(t_end_pose_cuda) / 1000
            t_elapsed_all_cuda = t_start_all_cuda.elapsed_time(t_end_all_cuda) / 1000
            with open('timings.txt', 'a') as fid:
                fid.write('{:10f}\t{:10f}\t{:10f}\t{:10f}\t{:10f}\n'.format(
                    t_elapsed_pp_cuda, t_elapsed_enc_cuda, t_elapsed_att_cuda,
                    t_elapsed_pose_cuda, t_elapsed_all_cuda
                ))

        outputs = {
            'src_feat_un': src_feats_un,
            'tgt_feat_un': tgt_feats_un,
            'src_feat': src_feats_list,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_list,  # List(B) of (N_pred, N_tgt, D)

            'src_kp': src_xyz_c,
            'src_kp_warped': src_corr_list,
            'tgt_kp': tgt_xyz_c,
            'tgt_kp_warped': tgt_corr_list,

            'src_mask_kp': src_mask_xyz,
            'src_mask_kp_warped': src_mask_corr,
            'tgt_mask_kp': tgt_mask_xyz,
            'tgt_mask_kp_warped': tgt_mask_corr,

            'src_overlap': src_overlap_list,
            'tgt_overlap': tgt_overlap_list,

            'pose': pred_pose_weighted,

            'is_train': is_train,
        }
        return outputs

    def compute_loss(self, pred, batch):
        B = pred['pose'].shape[0]
        losses = {}
        kpconv_meta = batch['kpconv_meta']
        pose_gt = batch['pose']
        p = len(kpconv_meta['stack_lengths']) - 1  # coarsest level
        batch['overlap_pyr'] = compute_overlaps(batch)
        src_overlap_p, tgt_overlap_p = \
            split_src_tgt(batch['overlap_pyr'][f'pyr_{p}'], kpconv_meta['stack_lengths'][p])
        all_overlap_pred = torch.cat(pred['src_overlap'] + pred['tgt_overlap'], dim=-2)
        all_overlap_gt = batch['overlap_pyr'][f'pyr_{p}']
        for i in self.cfg.overlap_loss_on:
            losses[f'overlap_{i}'] = self.overlap_criterion(all_overlap_pred[i, :, 0], all_overlap_gt)
        for i in self.cfg.feature_loss_on:
            losses[f'feature_{i}'] = self.feature_criterion(
                [s[i] for s in pred['src_feat']],
                [t[i] for t in pred['tgt_feat']],
                se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
            )
        losses['feature_un'] = self.feature_criterion_un(
            pred['src_feat_un'],
            pred['tgt_feat_un'],
            se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
        )
        for i in self.cfg.corr_loss_on:
            src_corr_loss, src_corr_err = self.corr_criterion(
                pred['src_kp'],
                [w[i] for w in pred['src_kp_warped']],
                batch['pose'],
                overlap_weights=src_overlap_p
            )
            tgt_corr_loss, tgt_corr_err = self.corr_criterion(
                pred['tgt_kp'],
                [w[i] for w in pred['tgt_kp_warped']],
                torch.stack([se3_inv(p) for p in batch['pose']]),
                overlap_weights=tgt_overlap_p
            )
            losses[f'corr_{i}'] = src_corr_loss + tgt_corr_loss


        if pred['is_train']:
            src_mask_corr_loss = self.chamfer_loss(
                pred['src_mask_kp'], pred['src_mask_kp_warped']
            )
            tgt_mask_corr_loss = self.chamfer_loss(
                pred['tgt_mask_kp'], pred['tgt_mask_kp_warped']
            )
            losses['mask_corr_0'] = src_mask_corr_loss + tgt_mask_corr_loss

        debug = False  # Set this to true to look at the registration result
        if debug:
            b = 0
            o = -1  # Visualize output of final transformer layer
            visualize_registration(batch['src_xyz'][b], batch['tgt_xyz'][b],
                                   torch.cat([pred['src_kp'][b], pred['src_kp_warped'][b][o]], dim=1),
                                   correspondence_conf=torch.sigmoid(pred['src_overlap'][b][o])[:, 0],
                                   pose_gt=pose_gt[b], pose_pred=pred['pose'][o, b])

        losses['total'] = torch.sum(
            torch.stack([(losses[k] * self.weight_dict[k]) for k in losses]))

        return losses


class CorrespondenceDecoder(nn.Module):
    def __init__(self, d_embed, use_pos_emb, pos_embed=None, num_neighbors=0):
        super().__init__()

        assert use_pos_emb is False or pos_embed is not None, \
            'Position encoder must be supplied if use_pos_emb is True'

        self.use_pos_emb = use_pos_emb
        self.pos_embed = pos_embed
        self.q_norm = nn.LayerNorm(d_embed)

        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.conf_logits_decoder = nn.Linear(d_embed, 1)
        self.num_neighbors = num_neighbors
        self.conf_match_decoder = nn.Linear(d_embed+6, 1)
        self.coor_mlp_ = nn.Sequential(
            nn.Linear(d_embed+6, d_embed+6),
            nn.ReLU(),
            nn.Linear(d_embed+6, d_embed+6),
            nn.ReLU(),
            nn.Linear(d_embed+6, 3)
        )

    def simple_attention(self, query, key, value, key_padding_mask=None):
        """Simplified single-head attention that does not project the value:
        Linearly projects only the query and key, compute softmax dot product
        attention, then returns the weighted sum of the values

        Args:
            query: ([N_pred,] Q, B, D)
            key: ([N_pred,] S, B, D)
            value: (S, B, E), i.e. dimensionality can be different
            key_padding_mask: (B, S)

        Returns:
            Weighted values (B, Q, E)
        """

        q = self.q_proj(query) / math.sqrt(query.shape[-1])
        k = self.k_proj(key)

        attn = torch.einsum('...qbd,...sbd->...bqs', q, k)  # (B, N_query, N_src)

        if key_padding_mask is not None:
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(key_padding_mask, float('-inf'))
            attn = attn + attn_mask[:, None, :]  # ([N_pred,] B, Q, S)

        if self.num_neighbors > 0:
            neighbor_mask = torch.full_like(attn, fill_value=float('-inf'))
            haha = torch.topk(attn, k=self.num_neighbors, dim=-1).indices
            neighbor_mask[:, :, haha] = 0
            attn = attn + neighbor_mask

        attn = torch.softmax(attn, dim=-1)

        attn_out = torch.einsum('...bqs,...sbd->...qbd', attn, value)

        return attn_out

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3)
            tgt_xyz: List of ([N_pred,] N_tgt, 3)

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)
        assert src_xyz_padded.shape[:-1] == src_feats_padded.shape[-3:-1] and \
               tgt_xyz_padded.shape[:-1] == tgt_feats_padded.shape[-3:-1]

        if self.use_pos_emb:
            both_xyz_packed = torch.cat(src_xyz + tgt_xyz)
            slens = list(map(len, src_xyz)) + list(map(len, tgt_xyz))
            src_pe, tgt_pe = split_src_tgt(self.pos_embed(both_xyz_packed), slens)
            src_pe_padded, _, _ = pad_sequence(src_pe)
            tgt_pe_padded, _, _ = pad_sequence(tgt_pe)
        src_feats2 = src_feats_padded + src_pe_padded if self.use_pos_emb else src_feats_padded
        tgt_feats2 = tgt_feats_padded + tgt_pe_padded if self.use_pos_emb else tgt_feats_padded
        src_corr = self.simple_attention(src_feats2, tgt_feats2, pad_sequence(tgt_xyz)[0],
                                         tgt_key_padding_mask)
        tgt_corr = self.simple_attention(tgt_feats2, src_feats2, pad_sequence(src_xyz)[0],
                                         src_key_padding_mask)

        num_pred = src_feats_padded.shape[0]

        src_feats2 = torch.cat([src_feats_padded,pad_sequence(src_xyz)[0].expand(num_pred,-1,-1,-1),src_corr],dim=-1)
        tgt_feats2 = torch.cat([tgt_feats_padded,pad_sequence(tgt_xyz)[0].expand(num_pred,-1,-1,-1),tgt_corr],dim=-1)

        src_bias = self.coor_mlp_(src_feats2)
        tgt_bias = self.coor_mlp_(tgt_feats2)

        src_corr = src_bias + src_corr
        tgt_corr = tgt_bias + tgt_corr

        src_feats2 = torch.cat([src_feats_padded,pad_sequence(src_xyz)[0].expand(num_pred,-1,-1,-1),src_corr],dim=-1)
        tgt_feats2 = torch.cat([tgt_feats_padded,pad_sequence(tgt_xyz)[0].expand(num_pred,-1,-1,-1),tgt_corr],dim=-1)

        src_match = self.conf_match_decoder(src_feats2)
        tgt_match = self.conf_match_decoder(tgt_feats2)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)
        src_match_list = unpad_sequences(src_match, src_lens)
        tgt_match_list = unpad_sequences(tgt_match, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list, src_match_list, tgt_match_list

class TemperatureNet(nn.Module):
    def __init__(self, d_embed, temp_factor):
        super(TemperatureNet, self).__init__()
        self.n_emb_dims = d_embed
        self.temp_factor = temp_factor
        self.nn = nn.Sequential(nn.Linear(self.n_emb_dims, 128),
                                nn.LayerNorm(128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 128),
                                nn.LayerNorm(128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 128),
                                nn.LayerNorm(128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 1),
                                nn.LeakyReLU())
        self.feature_disparity = None

    def forward(self, *input):

        src_embedding = input[0]
        tgt_embedding = input[1]
        src_embedding = src_embedding.mean(dim=1)
        tgt_embedding = tgt_embedding.mean(dim=1)

        residual = torch.abs(src_embedding-tgt_embedding)

        self.feature_disparity = residual
        Temperature = torch.clamp(self.nn(residual), 1.0/self.temp_factor, 1.0*self.temp_factor)
        return Temperature

class CorrespondenceRegressor(nn.Module):

    def __init__(self, d_embed):
        super().__init__()

        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3)
        )
        self.conf_logits_decoder = nn.Linear(d_embed, 1)

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3). Ignored
            tgt_xyz: List of ([N_pred,] N_tgt, 3). Ignored

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)
        src_corr = self.coor_mlp(src_feats_padded)
        tgt_corr = self.coor_mlp(tgt_feats_padded)

        num_pred = src_feats_padded.shape[0]

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list


class MaskRegressor(nn.Module):

    def __init__(self, d_embed, group_size):
        super().__init__()

        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3 * group_size)
        )

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3). Ignored
            tgt_xyz: List of ([N_pred,] N_tgt, 3). Ignored

        Returns:

        """
        _, g_src, B, _ = src_feats_padded.shape
        _, g_tgt, B, _ = tgt_feats_padded.shape

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)
        src_corr = self.coor_mlp(src_feats_padded).transpose(1, 2)
        tgt_corr = self.coor_mlp(tgt_feats_padded).transpose(1, 2)

        src_corr = src_corr.reshape(g_src*B, -1, 3)
        tgt_corr = tgt_corr.reshape(g_tgt*B, -1, 3)

        return src_corr, tgt_corr