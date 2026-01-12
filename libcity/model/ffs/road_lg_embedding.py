import torch.nn as nn
import torch
import math
import torch.nn.functional as F
# from libcity.model.bertlm.gat import GAT
from libcity.model.ffs import RoadLG
from libcity.model.bertlm.positional_embedding import PositionalEmbedding


class RoadLG_Embedding(nn.Module):

    def __init__(self,
                 d_model,
                 roadgm_d_model,
                 roadgm_add_layer_norm,
                 roadgm_add_batch_norm,
                 roadgm_gat_num_heads_per_layer,
                 roadgm_gat_num_features_per_layer,
                 roadgm_gat_bias,
                 roadgm_gat_dropout,
                 roadgm_gat_avg_last,
                 roadgm_gat_load_trans_prob,
                 roadgm_gat_add_skip_connection,
                 roadgm_mamba_attn_dropout,
                 dropout=0.1, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, node_fea_dim=10, add_gat=True,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        '''

        Args:
            #------------------------------Transformer Encoder Parameters-----------------
            d_model: embedding dim 即 d_model
            dropout: dropout ratio
            add_time_in_day:
            add_day_in_week:
            add_pe:
            #------------------------------GAT Parameters----------------------------------
            node_fea_dim:
            add_gat:
            gat_heads_per_layer:
            gat_features_per_layer:
            gat_dropout:
            load_trans_prob:
            avg_last:
        '''
        """
        official notes here
        Args:
            vocab_size: total vocab size
            d_model: embedding size of token embedding
            dropout: dropout rate
        """
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_pe = add_pe
        self.add_gat = add_gat

        # 对路段做embedding
        if self.add_gat:
            self.token_embedding = RoadLG(
                in_feature=node_fea_dim,
                d_model=d_model,
            )
        if self.add_pe:
            # 绝对位置嵌入
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, position_ids=None, graph_dict=None):
        """
        Args:
            sequence: (B, T, F) [loc, ts, mins, weeks, usr]
            position_ids: (B, T) or None
            graph_dict(dict): including:
                node_features: (vocab_size, road_fea_dim)
                edge_index: (2, E)
                loc_trans_prob: (E, 1)
        Returns:
            (B, T, d_model)

        """
        # T means seq_len
        if self.add_gat:
            x = self.token_embedding(node_features=graph_dict['node_features'],
                                     edge_index_input=graph_dict['edge_index'],
                                     edge_prob_input=graph_dict['edge_index_trans_prob'],
                                     x=sequence[:, :, 0].long())  # (B, T, d_model)
        if self.add_pe:
            # TODO note: x += 原地操作 inplace
            # x: (B,T,D), if position_embedding == None，pe(1,T,d_model), broad_cast
            # else not None, pe(B,T,d_mode)
            x += self.position_embedding(x, position_ids)  # (B, T, d_model)
        if self.add_time_in_day:
            # col 2: mins
            x += self.daytime_embedding(sequence[:, :, 2].long())  # (B, T, d_model)
        if self.add_day_in_week:
            # col 3: weeks
            x += self.weekday_embedding(sequence[:, :, 3].long())  # (B, T, d_model)
        return self.dropout(x)