from libcity.model.bertlm.gat import GATLayerImp3
from torch import nn
from libcity.model.ffs.gm_layer import GMLayer


class RoadGM(nn.Module):

    def __init__(
        self,
        in_feature,  # [int] Road-Node 的初始嵌入维度 (N_road, in_feature)
        d_model = 256, # [int] RoadGM每层的输出维度、及最后的输出维度; d_mdoel = num_heads_per_layer[0] * num_features_per_layer[0]
        add_layer_norm=True,  # 对于图结构(N, f)， per node的 layer norm更合适
        add_batch_norm=False,

        # -- gat
        num_heads_per_layer = [16, 16, 1], # [List] 每层头数
        num_features_per_layer = [16, 16, 256], # [List] 每层特征维度
        gat_bias=True, # [True] GAT Linear线性层添加偏置
        gat_dropout=0.1, # gat_dropout: [float] gat的dropout
        gat_avg_last=True, # [Bool] GAT 最后是否MeanPool
        gat_load_trans_prob=True, #
        gat_add_skip_connection=True,

        # -- mamba
        attn_dropout=0.1,
    ):
        '''

        Parameters
        ----------
        d_model: [int] RoadGM每层的输出维度、及最后的输出维度
        in_feature: [int] Road-Node 的初始嵌入维度 (N_road, in_feature)
        num_heads_per_layer: [List] 每层头数
        num_features_per_layer：[List] 每层特征维度
        add_skip_connection: [True] 默认添加残差
        add_layer_norm = False, 添加层归一
        add_batch_norm = True, 添加batch归一
        bias： [True] Linear线性层添加偏置
        gat_dropout: [float] gat的dropout
        attn_dropout：[float] xformer/mamba的dropout
        load_trans_prob: Bool 是否加载路段转移概率
        avg_last: 最后是否取平均
        '''
        super(RoadGM, self).__init__()

        self.d_model = d_model
        self.in_feature = in_feature # node_encoder_layer的输入维度
        self.in_heads = 1 # node_encoder_layer的输入头
        # roadgm gat layer
        self.num_heads_per_layer = num_heads_per_layer
        self.num_features_per_layer = num_features_per_layer
        num_of_layers = len(self.num_heads_per_layer) - 1
        assert self.d_model == (self.num_heads_per_layer[0] * self.num_features_per_layer[0])

        self.node_encoder_layer = GATLayerImp3(
            num_in_features= self.in_feature * self.in_heads,
            num_out_features= self.num_features_per_layer[0],
            num_of_heads= self.num_heads_per_layer[0],
            concat=True, # 最后一层在 concat
            activation= nn.ELU(),
            dropout_prob= gat_dropout,
            add_skip_connection= True,
            bias = True,
            load_trans_prob= gat_load_trans_prob,
        )

        # -- gm layers
        gm_layers = []
        for i in range(num_of_layers):
            if i == num_of_layers - 1: # 最后一层
                if gat_avg_last: # True
                    concat_input = False # avg
                else:
                    concat_input = True # concat
            else: # 其余
                concat_input = True # concat
            assert self.d_model == (self.num_features_per_layer[i]*self.num_heads_per_layer[i])
            layer = GMLayer(
                num_in_features= self.num_features_per_layer[i]*self.num_heads_per_layer[i], # d_model
                add_layer_norm=add_layer_norm,
                add_batch_norm=add_batch_norm,
                num_out_features= self.num_features_per_layer[i+1],
                num_of_heads= self.num_heads_per_layer[i+1],
                gat_bias = gat_bias,
                gat_dropout=gat_dropout,
                gat_avg_last =  (not concat_input), #
                gat_load_trans_prob= gat_load_trans_prob,
                gat_add_skip_connection= gat_add_skip_connection,

                attn_dropout= attn_dropout,
            )
            gm_layers.append(layer)
        self.gm_net = nn.Sequential(*gm_layers)

    def forward(self, node_features, edge_index_input, edge_prob_input, x): #
        data = (node_features, edge_index_input, edge_prob_input)
        # -- node encoding
        # node_features (N_road, f)
        data = self.node_encoder_layer(data)
        # -- gm_net
        (node_fea_emb, edge_index, edge_prob) = self.gm_net(data)  # (vocab_size, num_channels[-1]=D=d_model), (2, E)
        # -- x:(B,T)转 path_embedding_x:(B,T,d_model)
        batch_size, seq_len = x.shape
        node_fea_emb = node_fea_emb.expand((batch_size, -1, -1))  # (B, vocab_size, d_model)
        node_fea_emb = node_fea_emb.reshape(-1, self.d_model)  # (B * vocab_size, d_model)
        x = x.reshape(-1, 1).squeeze(1)  # (B * T,)
        out_node_fea_emb = node_fea_emb[x].reshape(batch_size, seq_len, self.d_model)  # (B, T, d_model)
        return out_node_fea_emb  # (B, T, d_model)




