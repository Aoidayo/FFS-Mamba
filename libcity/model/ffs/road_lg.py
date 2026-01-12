from mamba_ssm import Mamba
import torch
from torch_geometric.utils import scatter
from torch import nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul


class RoadLG(nn.Module):
    '''
    Road Local Global
    '''
    def __init__(self,
                 in_feature,
                 d_model=256,
                 khop=3,

                 # --
                 load_trans_prob=True,
                 dropout=0.1,
                 ):
        '''
        Structure
        ---------
        0、GAT initial： as node encoder
        1、LSEMba： Local State Evolution Mamba
        2、GCAMba：Global Context-Aware Mamba

        Parameters
        ----------
        in_feature
        d_model
        '''
        super(RoadLG, self).__init__()
        self.in_feature = in_feature
        self.d_model = d_model
        self.khop = khop

        # -- 1、node encoder GAT
        self.ne_in_features = in_feature
        self.ne_heads = 8
        self.ne_features = d_model // 8
        from libcity.model.bertlm.gat import GATLayerImp3
        self.node_encoder_layer = GATLayerImp3(
            num_in_features=self.ne_in_features,
            num_out_features=32,
            num_of_heads=8,
            concat=True,  # 最后一层在 concat
            activation=nn.ELU(),
            dropout_prob= dropout,
            add_skip_connection=True,
            bias=True,
            load_trans_prob= load_trans_prob,
        )

        # -- 2、LSEMba
        self.lsemba = Mamba(d_model = d_model, d_state=8, d_conv=4, expand=1)
        self.norm = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

    # def get_hop_sequence(self, x, edge_index, edge_prob=None):
    #     '''
    #
    #     Parameters
    #     ----------
    #     x   (vocab_size, f) 节点特征
    #     edge_index   (2, E) 有向边索引
    #     edge_prob    (E, 1) 边转移概率
    #
    #
    #     Returns
    #     -------
    #     seq_tensor : torch.Tensor
    #         形状 (N, num_hops + 1, feature_dim)
    #         包含 [0-hop, 1-hop, ..., k-hop] 的特征序列
    #     '''
    #     # 1. 初始化序列，放入原始特征 (0-hop)
    #     sequence = [x]
    #     curr_x = x
    #
    #     row, col = edge_index
    #     num_nodes = x.size(0)
    #
    #     # 2. 准备边权重 (Edge Weights)
    #     if edge_prob is None:
    #         # 如果没有提供概率，使用标准的 GCN 对称归一化: D^{-0.5} A D^{-0.5}
    #         # 计算节点的度
    #         deg = torch.bincount(row, minlength=num_nodes).float()
    #         deg_inv_sqrt = deg.pow(-0.5)
    #         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    #         # 计算边权重
    #         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    #     else:
    #         # 如果提供了转移概率，直接使用
    #         norm = edge_prob.squeeze()  # 确保形状是 (E,)
    #
    #     # 3. 循环生成多跳特征 (1 ~ num_hops)
    #     # 即使 self.use_gat=True，通常生成序列也建议用无参扩散，
    #     # 让 Mamba 去学习特征选择，这样更高效且显存占用低。
    #     from torch_sparse import matmul
    #     for _ in range(self.khop):
    #         # 执行稀疏矩阵乘法: X_next = A * X_curr
    #         # 消息构建: 源节点特征 (col) * 边权重
    #         msg = curr_x[col] * norm.view(-1, 1)
    #         # 消息聚合: 将消息累加到目标节点 (row)
    #         # scatter(src, index, dim, reduce='add')
    #         curr_x = scatter(msg, row, dim=0, reduce='add', dim_size=num_nodes)
    #         # (可选) 加入非线性激活，增加序列的非线性表达能力
    #         curr_x = F.relu(curr_x)
    #         # 加入序列
    #         sequence.append(curr_x)
    #
    #     # 4. 堆叠成 Mamba 需要的形状 [Batch, Length, Dim]
    #     # Length = num_hops + 1
    #     seq_tensor = torch.stack(sequence, dim=1)
    #     return seq_tensor

    from torch_sparse import SparseTensor

    def get_hop_sequence(self, x, edge_index, edge_prob=None):
        '''
        优化后的特征序列生成：使用 SparseTensor 矩阵乘法替代 scatter 循环
        '''
        num_nodes = x.size(0)
        row, col = edge_index

        # 1. 预处理权重：一次性计算完成
        if edge_prob is None:
            # 标准 GCN 归一化计算 (D^-0.5 * A * D^-0.5)
            deg = torch.bincount(row, minlength=num_nodes).float()
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = edge_prob.view(-1)

        # 2. 构建高性能稀疏邻接矩阵 (关键优化点)
        # SparseTensor 会自动在内部进行存储优化，加速 matmul
        adj_t = SparseTensor(row=row, col=col, value=norm,
                             sparse_sizes=(num_nodes, num_nodes))

        # 3. 初始化序列
        # 预先开辟空间比不断 append 列表更快
        sequence = torch.empty((num_nodes, self.khop + 1, x.size(-1)),
                               device=x.device, dtype=x.dtype)
        sequence[:, 0, :] = x

        curr_x = x

        # 4. 迭代扩散：此时 matmul 在 C++ 端高度并行化
        for k in range(1, self.khop + 1):
            # 执行 curr_x = adj_t @ curr_x
            # 相比于 scatter，matmul 显著减少了 Python 调用的 overhead
            curr_x = matmul(adj_t, curr_x, reduce='sum')

            if self.training:  # 训练时可以加入激活和 Dropout 防止平滑
                curr_x = F.relu(curr_x)

            sequence[:, k, :] = curr_x

        return sequence  # 形状 (N, num_hops + 1, feature_dim)



    def forward(self, node_features, edge_index_input, edge_prob_input, x):
        '''

        Parameters
        ----------
        node_features   (vocab_size, f)
        edge_index_input   (2, E)
        edge_prob_input    (E,1)
        x   (B, T, f), x[:,:,0].long() (B,T) 表示路段序列

        Returns
        -------

        '''
        data = (node_features, edge_index_input, edge_prob_input)
        # -- 1、node encoder
        data = self.node_encoder_layer(data)
        (node_fea_emb, edge_index, edge_prob) = data # (N,D)
        node_fea_emb_initial = node_fea_emb.clone().detach() # (N,D)
        # -- 2、lsemba
        # 不用这种方式，使用多阶邻域
        seq = self.get_hop_sequence(node_fea_emb, edge_index, edge_prob) # (N, L, D)
        out = self.lsemba(seq) # (N, L, D)
        filter_repr = out[:, -1, :] # (N, D)
        node_emb = self.out_proj(self.norm(filter_repr + node_fea_emb_initial))  # (N, D)
        # -- 3、compose
        batch_size, seq_len = x.shape
        node_emb = node_emb.expand((batch_size, -1, -1)) # (B, N, D)
        node_emb = node_emb.reshape(-1, self.d_model)  # (B*N, D)
        x = x.reshape(-1, 1).squeeze(1) # (B*T,)
        out_node_emb = node_emb[x].reshape(batch_size, seq_len, self.d_model)
        return out_node_emb

