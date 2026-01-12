import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import copy
import random
from logging import getLogger
from libcity.model.pm.mamba_gps_view import MambaGpsView
from libcity.model.pm import MambaRoadViewRoadGM


class FFSDP_Road(nn.Module):
    def __init__(self, config, road_gat_data):
        super(FFSDP_Road, self).__init__()
        self.config = config
        self.output_size = road_gat_data.get("vocab_size")

        self.d_model = self.config['mamba_gps_d_model']
        self._logger = getLogger()
        self._logger.info("🔧 构建FFS-Mamba的下游Adapter FFSDP_Road")

        self.road_view = MambaRoadViewRoadGM(self.config, road_gat_data)
        self.net =  nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.LayerNorm(self.d_model*2),
            nn.Linear(self.d_model*2, self.output_size),
        )
        # self.softmax = nn.Softmax(dim=-1)
        self.__load_pretrain()

    def forward(self, gps_X, gps_padding_mask):
        '''

        Parameters
        ----------
        gps_X (batch_size, len_gps, f_gps)
        gps_padding_mask (batch_size, len_gps)

        Returns
        -------
        pred (B,vocab_size)

        '''
        traj_emb, _ = self.road_view(gps_X, gps_padding_mask) # (B, D)
        pred = self.net(traj_emb) # (B, vocab_size)
        return pred

    def __load_pretrain(self):
        self._logger.info("⚙️ 加载预训练后的MambaGpsView")
        checkpoint = torch.load(self.config['pretrain_mamba_gps_view_ckpt'], map_location='cpu')
        self.gps_view.load_state_dict(checkpoint['model'].state_dict())
        self.gps_view.to(self.config['device'])
        # for param in self.gps_view.parameters():
        #     param.requires_grad = False