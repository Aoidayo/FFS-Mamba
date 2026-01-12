import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import copy
import random
from logging import getLogger

from libcity.model.bertlm.bert_contrastive_lm import MaskedLanguageModel
from libcity.model.ffs.mamba import TrajMixerModel
# from libcity.model.pm.bert_embedding import BERTEmbedding

from libcity.model.bertlm.BERT import TransformerBlock
from libcity.model.ffs.mamba2 import TrajMixerModel2



class TERMba(nn.Module):
    '''
    改写自RoadView


    '''
    def __init__(self, config, road_gat_data):
        super(TERMba, self).__init__()
        self.config = config

        # -- road gat
        self.vocab_size = road_gat_data['vocab_size']
        self.driver_num = road_gat_data['driver_num']
        self.node_fea_dim = road_gat_data['node_fea_dim']


        # -- model parameters in config
        self.transformer_d_model = self.config.get('transformer_d_model')
        self.transformer_n_layers = self.config.get('transformer_n_layers')
        self.transformer_attn_heads = self.config.get('transformer_attn_heads')
        # mlp中ffn的hidden size
        self.transformer_mlp_ratio = self.config.get('transformer_mlp_ratio', 4)
        # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        self.transformer_dropout = self.config.get('transformer_dropout', 0.1)
        # dropout of encoder block
        self.drop_path = self.config.get('drop_path', 0.3)
        self.transformer_attn_drop = self.config.get('transformer_attn_drop', 0.1)
        # pre-norm or post-norm, porto post
        self.type_ln = self.config.get('type_ln', 'post')
        # 是否遮盖未来时间步，双向transformer encoder比如Bert置为False
        self.future_mask = self.config.get('future_mask', False)
        # 在数据的开头添加cls，其实在dataset.BaseDataset中就已经使用
        # self.add_cls = self.config.get('add_cls', False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.add_minute_in_hour = self.config.get('add_minute_in_hour', True) # add_minute_in_hour
        self.add_time_in_day = self.config.get('add_time_in_day', True)
        self.add_day_in_week = self.config.get('add_day_in_week', True)
        self.add_pe = self.config.get('add_pe', True)
        self.add_gat = self.config.get('add_gat', True)
        self.gat_heads_per_layer = self.config.get('gat_heads_per_layer', [8, 16, 1])
        self.gat_features_per_layer = self.config.get('gat_features_per_layer', [16, 16, 256])
        self.gat_dropout = self.config.get('gat_dropout', 0.6)
        self.gat_avg_last = self.config.get('gat_avg_last', True)
        # self.load_trans_prob = self.config.get('load_trans_prob', False)
        self.add_temporal_bias = self.config.get('add_temporal_bias', True)
        self.temporal_bias_dim = self.config.get('temporal_bias_dim', 64)
        self.use_mins_interval = self.config.get('use_mins_interval', False) # 默认s间隔
        # -- -- RoadGM
        self.roadgm_d_model = self.config.get('roadgm_d_model')
        self.roadgm_add_layer_norm =self.config.get('roadgm_add_layer_norm')
        self.roadgm_add_batch_norm = self.config.get('roadgm_add_batch_norm')
        self.roadgm_gat_num_heads_per_layer = self.config.get('roadgm_gat_num_heads_per_layer')
        self.roadgm_gat_num_features_per_layer = self.config.get("roadgm_gat_num_features_per_layer")
        self.roadgm_gat_bias = self.config.get('roadgm_gat_bias')
        self.roadgm_gat_dropout = self.config.get("roadgm_gat_dropout")
        self.roadgm_gat_avg_last = self.config.get('roadgm_gat_avg_last')
        self.roadgm_gat_load_trans_prob = self.config.get('roadgm_gat_load_trans_prob')
        self.roadgm_gat_add_skip_connection = self.config.get('roadgm_gat_load_trans_prob')
        self.roadgm_mamba_attn_dropout = self.config.get('roadgm_mamba_attn_dropout')

        # -- computed parameters
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.transformer_feed_forward_hidden = self.transformer_d_model * self.transformer_mlp_ratio

        # -- RoadView
        self.mamba_road_d_model = self.config.get("mamba_road_d_model")
        self.mamba_road_embed_size = self.config.get('mamba_road_embed_size')
        self.mamba_road_use_mamba2 = self.config.get('mamba_road_use_mamba2')
        self.mamba_road_n_layer = self.config.get("mamba_road_n_layer")
        self.mamba_road_d_state = self.config.get('mamba_road_d_state')
        self.mamba_road_head_dim = self.config.get('mamba_road_head_dim')
        self.mamba_road_d_inner = self.config.get('mamba_road_d_inner')

        from libcity.model.pm.TERMba_LPN_GPR import TERMba_LPN_GPR
        self.embedding = TERMba_LPN_GPR(
           d_model=self.transformer_d_model,
           dropout=self.transformer_dropout,
           add_time_in_day=self.add_time_in_day,
           add_day_in_week=self.add_day_in_week,
           add_pe=self.add_pe,
           node_fea_dim=self.node_fea_dim,
           add_gat=self.add_gat,
           gat_heads_per_layer=self.gat_heads_per_layer,
           gat_features_per_layer=self.gat_features_per_layer,
           gat_dropout=self.gat_dropout,
           load_trans_prob=True,
           avg_last=self.gat_avg_last,
            # roadgm
            roadgm_d_model=  self.roadgm_d_model,
            roadgm_add_layer_norm = self.roadgm_add_layer_norm,
            roadgm_add_batch_norm = self.roadgm_add_batch_norm,
            roadgm_gat_num_heads_per_layer = self.roadgm_gat_num_heads_per_layer,
            roadgm_gat_num_features_per_layer = self.roadgm_gat_num_features_per_layer,
            roadgm_gat_bias = self.roadgm_gat_bias,
            roadgm_gat_dropout = self.roadgm_gat_dropout,
            roadgm_gat_avg_last = self.roadgm_gat_avg_last,
            roadgm_gat_load_trans_prob = self.roadgm_gat_load_trans_prob,
            roadgm_gat_add_skip_connection = self.roadgm_gat_add_skip_connection,
            roadgm_mamba_attn_dropout = self.roadgm_mamba_attn_dropout,
            vocab_size= self.vocab_size,
        )

        # -- transformer blocks
        from libcity.model.pm.mamba_gps.mamba import RoadMixerModel
        self.mamba_blocks = RoadMixerModel(
            d_model = self.mamba_road_d_model,
            n_layer = self.mamba_road_n_layer,
            device = self.device,
            dtype = torch.float32,
        )
        # mask_l, 预测span-mlm的road
        self.mask_l = MaskedLanguageModel(self.transformer_d_model, self.vocab_size)

    def reverse_padded_sequence(self, x, lengths):
        """
        x: (B, T, D)
        lengths: (B,) 记录每个序列的有效长度
        """
        x_rev = torch.zeros_like(x)
        batch_size = x.size(0)
        for i in range(batch_size):
            l = lengths[i]
            # 取出有效部分 -> 翻转 -> 填回 0...l
            x_rev[i, :l] = x[i, :l].flip(dims=[0])
        return x_rev

    def forward(self, x, padding_masks, batch_temporal_mat=None
                , graph_dict=None,
                output_hidden_states=False, output_attentions=False):
        """
        Args:
        -------
        x: (batch_size, seq_length, feat_dim) torch tensor of masked features and padded length (input)
        padding_masks: (batch_size, seq_length) boolean tensor,
            keep in the valid length, else invalid
            1 means keep vector at this position, 0 means padding
        batch_temporal_mat: (batch_size, seq_len, seq_len)
        graph_dict:
        output_hidden_states: False，是否返回hidden_states
        output_attentions: False, 是否返回attentions

        Returns:
        -------
        output: (batch_size, seq_length, feat_dim)
        """
        position_ids = None

        # embedding the indexed sequence to sequence of vectors
        # (B, T, d_model)
        embedding_output, global_embedding_output = self.embedding(
            sequence=x,
            position_ids=position_ids,
            graph_dict=graph_dict
        )


        # # padding_masks: (B,T,)
        # padding_masks_input = padding_masks.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (B, 1, T, T)
        # # running over multiple transformer blocks
        # # output_hidden_states: False, all_hidden_states = [ embedding_output:(B,T,D) ],
        # # output_attentions: False, all_attentions = []
        # all_hidden_states = [embedding_output] if output_hidden_states else None
        # all_self_attentions = [] if output_attentions else None
        # for transformer in self.transformer_blocks:
        #     # (B, T, d_model)
        #     embedding_output, attn_score = transformer.forward(
        #         # (B, T, d_model)
        #         x=embedding_output,
        #         # padding_masks_input = padding_masks.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (B, 1, T, T)
        #         padding_masks=padding_masks_input,
        #         future_mask=False,
        #         # output_attentions: False
        #         output_attentions=output_attentions,
        #         # (Batch_size, seq_len, seq_len)
        #         batch_temporal_mat=batch_temporal_mat)
        #     if output_hidden_states:
        #         all_hidden_states.append(embedding_output)
        #     if output_attentions:
        #         all_self_attentions.append(attn_score)

        # x_local = self.local_conv(embedding_output.permute(0, 2, 1)).permute(0, 2, 1)
        embedding_output = embedding_output # + x_local
        # embedding_output : (B, T, d_model)
        # -- forward
        out_fwd = self.mamba_blocks(
            embedding_output, global_embedding_output
        )
        out_fwd_padded = out_fwd * padding_masks.unsqueeze(-1) # (B, T, D) * (B, T, 1)
        valid_lengths = padding_masks.sum(dim=1).long() # (B,T) True保留，False遮盖 -> (B)

        # -- backward
        # input_rev = torch.flip(embedding_output, dims=[1])
        input_rev_clean = self.reverse_padded_sequence(embedding_output, valid_lengths)
        global_input_rev_clean = self.reverse_padded_sequence(global_embedding_output, valid_lengths)

        # padding_masks_rev = torch.flip(padding_masks, dims=[1])
        out_rev = self.mamba_blocks(input_rev_clean, global_input_rev_clean)
        out_rev_aligned = self.reverse_padded_sequence(out_rev, valid_lengths)
        out_rev_padded = out_rev_aligned * padding_masks.unsqueeze(-1)
        road_embedding_output = out_fwd + out_rev_aligned
        road_embedding_output_padded = out_fwd_padded + out_rev_padded

        # padding_masks的均值池化
        mask = padding_masks.unsqueeze(-1)  # (B, T, 1)
        valid_lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1) 有效长度 (防止除以0)
        masked_embeddings = road_embedding_output_padded * mask  # (B, T, D)
        mean_pooled = masked_embeddings.sum(dim=1) / valid_lengths  # (B, T, D) -> (B, D)

        a = self.mask_l(road_embedding_output_padded) # (B, T, vocab_size)
        return road_embedding_output, mean_pooled, a #(B, d_model), (B, T, vocab_size)