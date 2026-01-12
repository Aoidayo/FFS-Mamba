import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import copy
import random
from logging import getLogger
# 循环依赖引用问题，这里修改为延迟引入
# from libcity.model.pm import MambaRoadViewRoadGM
from libcity.model.pm import MambaRoadView
from libcity.model.pm.mamba_gps_view import MambaGpsView


# class CrossAttn(nn.Module):
#     def __init__(self, config):
#         super(CrossAttn, self).__init__()
#         self.d_model = 256
#         self.nhead=4
#         self.dropout=0.1
#         self.attn = nn.MultiheadAttention(
#             embed_dim=self.d_model,
#             num_heads=self.nhead,
#             dropout=self.dropout,
#             batch_first=True  # 输出维度保持 (B, L, D)
#         )
#
#         # ---- Gate: 让模型自己决定要不要融合GPS语义 ----
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(self.d_model * 2, self.d_model),
#             nn.ReLU(),
#             nn.Linear(self.d_model, 1),  # 每个 token 一个 gate scalar
#             nn.Sigmoid()
#         )
#         self.norm = nn.LayerNorm(self.d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(self.d_model, 4 * self.d_model),
#             nn.ReLU(),
#             nn.Linear(4 * self.d_model, self.d_model),
#         )
#         self.ffn_norm = nn.LayerNorm(self.d_model)
#
#
#     def forward(self, road_embedding, gps_embedding, road_padding_mask, gps_padding_mask):
#         '''
#
#         Parameters
#         ----------
#         road_embedding  (B,L_r,D)
#         gps_embedding   (B,L_g,D)
#         road_padding_mask   (B,L_r), 1保留，0遮盖
#         gps_padding_mask (B, L_g), 1保留，0遮盖
#
#         Returns
#         -------
#         road_fuse_gps: (B,L_r,D)
#         gps_fuse_road: (B,L_g,D)
#
#         '''
#
#         # ---- Cross Attention : Q = road, K = gps, V = gps ----
#         # 使用road聚合gps点的连续运动特征
#         # PyTorch 的 attn_mask 是 bool: True = mask 掉
#         # 而你的是 1=keep,0=mask，因此取反
#         gps_key_padding_mask = (gps_padding_mask == 0)
#         attn_output, _ = self.attn(
#             query=road_embedding,
#             key=gps_embedding,
#             value=gps_embedding,
#             key_padding_mask=gps_key_padding_mask  # (B, L_g)
#         )
#
#         # -- gate: 基于road和attn_output决定融合多少
#         gate_input = torch.cat([road_embedding, attn_output], dim=-1)
#         gate = self.gate_mlp(gate_input)  # (B, L_r, 1)
#         gated_attn = gate * attn_output  # (B, L_r, D)
#
#         # Residual + Norm
#         road = self.norm(road_embedding + gated_attn)
#         # FFN
#         ffn_output = self.ffn(road)
#         output = self.ffn_norm(road + ffn_output)
#         return output


def masked_mean_pooling(x, mask):
    """
    x: (B, L, D)
    mask: (B, L)  1 = keep, 0 = padding

    returns (B, D)
    """
    mask = mask.unsqueeze(-1)         # (B, L, 1)
    x = x * mask                      # 把 padding 位置全部置 0
    sum_x = x.sum(dim=1)              # (B, D)
    len_x = mask.sum(dim=1)           # (B, 1)

    # 防止除以 0
    len_x = len_x.clamp(min=1)

    return sum_x / len_x


class CrossAttn(nn.Module):
    def __init__(self, config=None):
        super(CrossAttn, self).__init__()
        self.config = config
        self.d_model = self.config.get("mamba_gps_d_model")
        self.nhead = 4
        self.dropout = 0.1

        # ========= Road ← GPS (你的原版) =========
        self.attn_rg = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.gate_rg = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.norm_rg = nn.LayerNorm(self.d_model)
        self.ffn_rg = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.ReLU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )
        self.ffn_norm_rg = nn.LayerNorm(self.d_model)

        # ========= GPS ← Road (反写的版本) =========
        self.attn_gr = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.gate_gr = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.norm_gr = nn.LayerNorm(self.d_model)
        self.ffn_gr = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.ReLU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )
        self.ffn_norm_gr = nn.LayerNorm(self.d_model)

    def forward(self, road_embedding, gps_embedding,
                road_padding_mask, gps_padding_mask):
        """
        Parameters
        ----------
        road_embedding: (B, L_r, D)
        gps_embedding:  (B, L_g, D)
        road_padding_mask: (B, L_r) 0遮盖，1保留
        gps_padding_mask:  (B, L_g)  0遮盖，1保留

        Returns
        -------
        road_fuse_gps: (B, L_r, D)
        gps_fuse_road: (B, L_g, D)
        """

        # ========== 1. Road ← GPS  (Q=road, K=gps, V=gps) ==========
        gps_kpm = (gps_padding_mask == 0) # attn中1遮盖，取反

        attn_rg, _ = self.attn_rg(
            query=road_embedding,
            key=gps_embedding,
            value=gps_embedding,
            key_padding_mask=gps_kpm
        )

        gate_rg_input = torch.cat([road_embedding, attn_rg], dim=-1)
        gate_rg = self.gate_rg(gate_rg_input)
        gated_rg = gate_rg * attn_rg

        road_mid = self.norm_rg(road_embedding + gated_rg)
        road_fuse_gps = self.ffn_norm_rg(road_mid + self.ffn_rg(road_mid)) # (B,L_r,D)

        # ========== 2. GPS ← Road  (Q=gps, K=road, V=road) ==========
        road_kpm = (road_padding_mask == 0)

        attn_gr, _ = self.attn_gr(
            query=gps_embedding,
            key=road_embedding,
            value=road_embedding,
            key_padding_mask=road_kpm
        )

        gate_gr_input = torch.cat([gps_embedding, attn_gr], dim=-1)
        gate_gr = self.gate_gr(gate_gr_input)
        gated_gr = gate_gr * attn_gr

        gps_mid = self.norm_gr(gps_embedding + gated_gr)
        gps_fuse_road = self.ffn_norm_gr(gps_mid + self.ffn_gr(gps_mid)) # (B,L_g,D)

        # -- 3、pooled by mask
        road_pooled = masked_mean_pooling(road_fuse_gps, road_padding_mask)  # (B, D)
        gps_pooled = masked_mean_pooling(gps_fuse_road, gps_padding_mask)  # (B, D)

        return road_fuse_gps, gps_fuse_road, road_pooled, gps_pooled



class MambaDp(nn.Module):
    def __init__(self, config, road_gat_data):
        '''MambaFuseView = MambaGpsView + MambaRoadView -> Contrast + SpanMlmLoss


        Parameters
        ----------
        config
        road_gat_data
        '''
        super(MambaDp, self).__init__()

        self.config = config
        self.device = config['device']
        self._logger = getLogger()
        self.dp_type = config.get("dp_type",None)
        self.d_model = self.config['mamba_gps_d_model']
        self.road_output_size = road_gat_data.get("vocab_size",None)
        self.gps_output_size = 2

        # -- Model Initialize
        # -- --  RoadView Model
        # from libcity.model.ffs import RoadView
        # self.road_view = RoadView(self.config, road_gat_data)
        # --------------------- PATHTST -----------------------
        # from libcity.model.ffs.patchtst import PatchTSTRoadView,PatchTSTRoadConfig
        # cfg = PatchTSTRoadConfig(
        #     vocab_size=6253,
        #     min_seq_len=11,
        #     d_model=256,
        #     patch_length=4,
        #     patch_stride=2,
        #     num_layers=4,
        #     num_heads=8,
        #     dropout=0.1,
        #     add_gat=False,
        #     add_pe=True,
        #     add_time_in_day=True,
        #     add_day_in_week=True,
        #     node_fea_dim=10,
        #     roadgm_d_model=256,
        # )
        # self.road_view = PatchTSTRoadView(cfg)
        # --------------------- PATHTST -----------------------
        # --- TERMba
        from libcity.model.ffs.TERMba import TERMba
        self.road_view = TERMba(self.config, road_gat_data)
        # --- TERMba
        # ----------------- MambaRoadView -----------------------
        # from libcity.model.pm import MambaRoadView
        # self.road_view = MambaRoadView(config, road_gat_data)
        # from libcity.model.pm import MambaRoadViewRoadGM
        # self.road_view = MambaRoadViewRoadGM(config, road_gat_data)
        # ----------------- MambaRoadView --------------------



        # -- -- GpsView Model
        self.gps_view = MambaGpsView(config)
        # -- -- Cross Attn
        self.cross_attn = CrossAttn(config)
        # -- --  Cross
        # 可学习的缩放模型输出的 logits（未归一化的预测），以控制不同模态之间的相似度度量。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy = nn.CrossEntropyLoss()
        # -- -- Model Adapter Initialize
        self.adapter_output_size = self.road_output_size  if self.dp_type == 'road' else self.gps_output_size
        self.adapter = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.LayerNorm(self.d_model*2),
            nn.Linear(self.d_model*2, self.adapter_output_size),
            nn.LogSoftmax(dim=-1)
        )

        # -- load ckpt
        self.pretrain_mamba_ckpt_path = self.config.get("pretrain_mamba_ckpt", None)
        # self.pretrain_mamba_road_view_ckpt_path = self.config.get('pretrain_mamba_road_view_ckpt', None)
        if self.pretrain_mamba_ckpt_path is not None:
            self.__load_road_view()
        else:
            self._logger.info("⚙️ 不使用预训练后的MambaRoadView")
        # assert self.pretrain_mamba_road_view_ckpt_path is not None, "pretrain_mamba_road_view_ckpt_path must be defined"


    def __load_road_view(self):
        self._logger.info("⚙️ 加载预训练后的MambaRoadView,MambaGpsView,CrossAttn")
        checkpoint = torch.load(self.pretrain_mamba_ckpt_path, map_location='cpu')
        self.road_view.load_state_dict(checkpoint['model'].road_view.state_dict())
        self.road_view.to(self.config['device'])
        self.gps_view.load_state_dict(checkpoint['model'].gps_view.state_dict())
        self.gps_view.to(self.config['device'])
        self.cross_attn.load_state_dict(checkpoint['model'].cross_attn.state_dict())
        self.cross_attn.to(self.config['device'])
        # -- 冻结参数
        # self._logger.info("🧊 冻结MambaRoadView中的所有参数")
        # for param in self.road_view.parameters():
        #     param.requires_grad = False

    def forward(
            self,
            gps_X, gps_padding_mask,
            road_X, padding_masks, batch_temporal_mat,
            # aug_gps_X, aug_gps_padding_mask,
            graph_dict):
        '''

        Parameters
        ----------
        gps_X   (B, len_gps, f_gps)
        gps_padding_mask    (B, len_gps)  0表示遮盖
        road_X  (B, len_road, f_road)
        padding_masks   (B, len_road, len_road)
        batch_temporal_mat  (B, len_road, len_road)

        aug_gps_X   (B, len_gps, f_gps)
        aug_gps_padding_mask    (B, len_gps)

        graph_dict

        Returns
        -------
        road_fuse_pooled:  (B, D)
        gps_fuse_pooled:  (B, D)
        road_mask_l: (B, T, Vocab_size)
        road_mean_pooled: (B,D) 由roadview直接输出
        gps_mean_pooled: (B,D) 由gps view直接输出


        '''
        embedding_output, road_mean_pooled, road_mask_l = self.road_view(road_X, padding_masks, batch_temporal_mat, graph_dict) # (B, D)
        gps_embedding_output, gps_mean_pooled = self.gps_view(gps_X, gps_padding_mask)  # (B, D)
        # aug_gps_embedding_output, aug_gps_mean_pooled = self.gps_view(aug_gps_X, aug_gps_padding_mask)
        # aug_gps_embedding = self.gps_view(aug_gps_X, aug_gps_padding_mask) # (B,D)
        road_fuse_gps, gps_fuse_road, road_fuse_pooled, gps_fuse_pooled = self.cross_attn(
            road_embedding=embedding_output, gps_embedding=gps_embedding_output,
            road_padding_mask=padding_masks, gps_padding_mask = gps_padding_mask
        ) # (B,L_r,D), (B,L_g,D), (B,D), (B,D)

        if self.dp_type == 'gps':
            gps_dp_pred = self.adapter(gps_fuse_pooled) # (B,2)
            return gps_dp_pred
        elif self.dp_type == 'road':
            road_dp_pred = self.adapter(embedding_output) # (B,Vocab_size)
            return road_dp_pred

        # return road_fuse_gps, gps_fuse_road, road_fuse_pooled, gps_fuse_pooled, \
        #     road_mask_l, road_mean_pooled, gps_mean_pooled # , aug_gps_mean_pooled # (B,D), (B,D)

    def forward_on_span_road(
            self,
            gps_X, gps_padding_mask,
            road_X, padding_masks, batch_temporal_mat,
            # aug_gps_X, aug_gps_padding_mask,
            graph_dict):
        '''

        Parameters
        ----------
        gps_X   (B, len_gps, f_gps)
        gps_padding_mask    (B, len_gps)  0表示遮盖
        road_X  (B, len_road, f_road)
        padding_masks   (B, len_road, len_road)
        batch_temporal_mat  (B, len_road, len_road)

        aug_gps_X   (B, len_gps, f_gps)
        aug_gps_padding_mask    (B, len_gps)

        graph_dict

        Returns
        -------
        road_fuse_pooled:  (B, D)
        gps_fuse_pooled:  (B, D)
        road_mask_l: (B, T, Vocab_size)
        road_mean_pooled: (B,D) 由roadview直接输出
        gps_mean_pooled: (B,D) 由gps view直接输出


        '''
        embedding_output, road_mean_pooled, road_mask_l = self.road_view(road_X, padding_masks, batch_temporal_mat, graph_dict) # (B, D)
        gps_embedding_output, gps_mean_pooled = self.gps_view(gps_X, gps_padding_mask)  # (B, D)
        # aug_gps_embedding_output, aug_gps_mean_pooled = self.gps_view(aug_gps_X, aug_gps_padding_mask)
        # aug_gps_embedding = self.gps_view(aug_gps_X, aug_gps_padding_mask) # (B,D)
        road_fuse_gps, gps_fuse_road, road_fuse_pooled, gps_fuse_pooled = self.cross_attn(
            road_embedding=embedding_output, gps_embedding=gps_embedding_output,
            road_padding_mask=padding_masks, gps_padding_mask = gps_padding_mask
        ) # (B,L_r,D), (B,L_g,D), (B,D), (B,D)

        # padding_masks的均值池化
        road_fuse_gps_ada = self.adapter(road_fuse_gps)  # (B, T, vocab_size)
        return road_fuse_gps_ada # (B, T, Vocab_size)

        # if self.dp_type == 'gps':
        #     gps_dp_pred = self.adapter(gps_fuse_pooled) # (B,2)
        #     return gps_dp_pred
        # elif self.dp_type == 'road':
        #     road_dp_pred = self.adapter(road_fuse_pooled) # (B,Vocab_size)
        #     return road_dp_pred

        # return road_fuse_gps, gps_fuse_road, road_fuse_pooled, gps_fuse_pooled, \
        #     road_mask_l, road_mean_pooled, gps_mean_pooled # , aug_gps_mean_pooled # (B,D), (B,D)

if __name__ == '__main__':
    print("==== Running CrossAttn Test ====")

    B = 2  # batch size
    L_r = 5  # road sequence length
    L_g = 7  # gps sequence length
    D = 256  # embedding dim

    cross_attn = CrossAttn()

    # ---- 随机嵌入 ----
    road_embedding = torch.randn(B, L_r, D)
    gps_embedding = torch.randn(B, L_g, D)

    # ---- 随机 mask：每条轨迹不同长度 ----
    road_padding_mask = torch.tensor([
        [1, 1, 1, 1, 0],  # 最后一个为 padding
        [1, 1, 0, 0, 0]
    ])

    gps_padding_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0],  # 两个 padding
        [1, 1, 1, 0, 0, 0, 0]
    ])

    # ---- forward ----
    road_fuse_gps, gps_fuse_road, road_pooled, gps_pooled = cross_attn(road_embedding, gps_embedding,
                        road_padding_mask, gps_padding_mask)

    print("output shape =", road_fuse_gps.shape)

    # ---- Assertions for sanity check ----
    assert road_fuse_gps.shape == (B, L_r, D), "Output shape mismatch!"
    assert not torch.isnan(road_fuse_gps).any(), "Output has NaN!"
    assert not torch.isinf(road_fuse_gps).any(), "Output has Inf!"

    # ---- masking test: ensure padding positions 仍然有合理数值 ----
    padded_positions = (road_padding_mask == 0)
    print("Padding positions in road_mask:", padded_positions)
    print("Output on padding pos:", road_fuse_gps[padded_positions])

    print("Test passed!")
    print("================================")