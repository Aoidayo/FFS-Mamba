import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import copy
import random
from logging import getLogger
from libcity.model.pm.mamba_gps_view import MambaGpsView


class FFSTTE(nn.Module):
    def __init__(self, config):
        super(FFSTTE, self).__init__()
        self.config = config

        self.d_model = self.config['mamba_gps_d_model']
        self._logger = getLogger()
        self._logger.info("🔧 构建FFS-Mamba的下游Adapter FFSTTE")

        self.gps_view = MambaGpsView(self.config)
        self.linear = nn.Linear(self.d_model, 1)
        self.__load_pretrain()

    def forward(self, gps_X, gps_padding_mask):
        '''

        Parameters
        ----------
        gps_X (batch_size, len_gps, f_gps)
        gps_padding_mask (batch_size, len_gps)

        Returns
        -------
        eta_pred (B,1)

        '''
        traj_emb = self.gps_view(gps_X, gps_padding_mask)
        eta_pred = self.linear(traj_emb) # (B,1)
        return eta_pred

    def __load_pretrain(self):
        self._logger.info("⚙️ 加载预训练后的MambaGpsView")
        checkpoint = torch.load(self.config['pretrain_mamba_gps_view_ckpt'], map_location='cpu')
        self.gps_view.load_state_dict(checkpoint['model'].state_dict())
        self.gps_view.to(self.config['device'])