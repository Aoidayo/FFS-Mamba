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
from libcity.model.pm.mamba_gps_view import MambaGpsView


class MambaFuseAugView(nn.Module):
    def __init__(self, config, road_gat_data):
        '''MambaFuseView = MambaGpsView + MambaRoadView -> Contrast + SpanMlmLoss


        Parameters
        ----------
        config
        road_gat_data
        '''
        super(MambaFuseAugView, self).__init__()

        self.config = config
        self.device = config['device']
        self._logger = getLogger()
        self.pretrain_mamba_road_view_ckpt_path = self.config.get('pretrain_mamba_road_view_ckpt', None)

        # -- RoadView Model
        from libcity.model.pm import MambaRoadViewRoadGM
        self.road_view = MambaRoadViewRoadGM(config, road_gat_data)
        assert self.pretrain_mamba_road_view_ckpt_path is not None, "pretrain_mamba_road_view_ckpt_path must be defined"
        self.__load_road_view()

        # -- GpsView Model
        self.gps_view = MambaGpsView(config)

        # -- Cross
        # 可学习的缩放模型输出的 logits（未归一化的预测），以控制不同模态之间的相似度度量。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy = nn.CrossEntropyLoss()


    def __load_road_view(self):
        self._logger.info("⚙️ 加载预训练后的MambaRoadView")
        checkpoint = torch.load(self.pretrain_mamba_road_view_ckpt_path, map_location='cpu')
        self.road_view.load_state_dict(checkpoint['model'].state_dict())
        self.road_view.to(self.config['device'])
        # -- 冻结参数
        self._logger.info("🧊 冻结MambaRoadView中的所有参数")
        for param in self.road_view.parameters():
            param.requires_grad = False

    def forward(
            self,
            gps_X, gps_padding_mask,
            road_X, padding_masks, batch_temporal_mat,
            aug_gps_X, aug_gps_padding_mask,
            graph_dict):
        '''

        Parameters
        ----------
        gps_X   (B, len_gps, f_gps)
        gps_padding_mask    (B, len_gps)
        road_X  (B, len_road, f_road)
        padding_masks   (B, len_road, len_road)
        batch_temporal_mat  (B, len_road, len_road)

        aug_gps_X   (B, len_gps, f_gps)
        aug_gps_padding_mask    (B, len_gps)

        graph_dict

        Returns
        -------
        road_embedding  (B, D)
        gps_embedding  (B, D)

        '''
        road_embedding, _ = self.road_view(road_X, padding_masks, batch_temporal_mat, graph_dict) # (B, D)
        gps_embedding = self.gps_view(gps_X, gps_padding_mask)  # (B, D)
        aug_gps_embedding = self.gps_view(aug_gps_X, aug_gps_padding_mask) # (B,D)

        return road_embedding, gps_embedding, aug_gps_embedding

