# gat_pyg.py  ← 这次是真的最终版
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GATLayerPyG(MessagePassing):
    def __init__(self,
                 num_in_features,
                 num_out_features,
                 num_of_heads,
                 concat=True,
                 dropout_prob=0.6,
                 add_skip_connection=True,
                 bias=True,
                 activation=nn.ELU(),
                 load_trans_prob=True):
        super().__init__(aggr='add', node_dim=0)

        self.num_in_features     = num_in_features
        self.num_out_features    = num_out_features
        self.num_of_heads        = num_of_heads
        self.concat              = concat
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob     = load_trans_prob
        self.activation          = activation

        self.dropout = nn.Dropout(dropout_prob)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.lin = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # 残差投影：只有维度不匹配时才创建
        if add_skip_connection and (num_in_features != num_of_heads * num_out_features):
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.skip_proj = None

        self.att_src = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if load_trans_prob:
            self.lin_edge = nn.Linear(1, num_of_heads * num_out_features, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        else:
            self.lin_edge = None
            self.att_edge = None

        self.to_e = nn.Linear(num_of_heads, num_of_heads, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(num_of_heads * num_out_features if concat else num_out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.skip_proj is not None:
            nn.init.xavier_uniform_(self.skip_proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_prob=None):
        N = x.size(0)

        # 1. 两次 dropout + 线性变换
        x = self.dropout(x)
        h = self.lin(x).view(-1, self.num_of_heads, self.num_out_features)
        h = self.dropout(h)                                   # 关键第二次 dropout

        # 2. 消息传递
        out = self.propagate(edge_index, x=(h, h), edge_attr=edge_prob)

        # 3. 残差连接 —— 严格按照原实现逻辑！！
        if self.add_skip_connection:
            if self.skip_proj is not None:
                # 维度不匹配 → 用投影
                residual = self.skip_proj(x).view(-1, self.num_of_heads, self.num_out_features)
            else:
                # 维度匹配（如 512 → 512）→ 直接加原始特征（unsqueeze 广播）
                residual = x.unsqueeze(1)                     # [N,FIN] → [N,1,FIN]
                # FIN == heads * FOUT, 所以可以广播
            out = out + residual

        # 4. concat / mean
        out = out.flatten(1) if self.concat else out.mean(dim=1)

        # 5. bias + activation
        if self.bias is not None:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)

        return out, edge_index, edge_prob

    def message(self, x_i, x_j, edge_attr, index):
        alpha = (x_i * self.att_src).sum(-1) + (x_j * self.att_dst).sum(-1)

        if self.load_trans_prob and edge_attr is not None:
            e = self.lin_edge(edge_attr).view(-1, self.num_of_heads, self.num_out_features)
            e = self.dropout(e)
            alpha = alpha + (e * self.att_edge).sum(-1)

        alpha = self.leaky_relu(self.to_e(alpha))
        alpha = softmax(alpha, index, num_nodes=x_i.size(0))
        alpha = self.dropout(alpha).unsqueeze(-1)

        return x_j * alpha