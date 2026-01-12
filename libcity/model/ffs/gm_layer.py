import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index
from typing import List
from libcity.model.bertlm.gat import GATLayerImp3


def lexsort_with_graphbatch(
        keys: List[Tensor],
        dim: int = -1,
        descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    按多个 key 做“字典序排序”（lexicographic sort），
    从最后一个 key 开始作为主排序键，前面的 key 是次级排序键。
    返回的是排序后的索引，而不是排序后的值。

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key. 最后一个key时主sort key
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`) 默认升序

    Return:
        List[torch.Tensor: longTensor]: Shape(keys[0].shape[0])

    Example:
        keys: [key1, key2]
            key1:  [2,5,1]
            key2:  [0, 1, 1], 一般key2表示所属图

            先按 key2 排序（主要排序键）
            key2 相同的再按 key1 排序（次级排序键）

            得到字典序排序的索引输出就是 [0, 2, 1]

        如果keys只有一个的话，就是直接按key1的排序输出索引

    """
    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out

def lexsort_singlegraph(
    keys,
    dim=-1,
    descending= False,
) -> Tensor:
    '''

    Parameters
    ----------
    keys [2,5,1]
    dim -1
    descending False

    Returns
    -------
    out_sort_index [2, 0, 1]
    '''
    out = keys.argsort(dim=dim, descending=descending, stable=True)
    return out


def permute_within_batch(batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices

class GMLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
            x: mamba
    """
    def __init__(
        self,
        # mamba / gat, num_in_features = d_mdoel
        num_in_features, # d_model = 256, num_in_features
        add_layer_norm,
        add_batch_norm,
        # -- gat
        num_of_heads, # num_in_features = num_of_heads * num_out_features
        num_out_features,
        gat_bias,
        gat_dropout,
        gat_avg_last, # True 表示均值池化, 是否是最后一层
        gat_load_trans_prob, # gat使用转移概率计算
        gat_add_skip_connection,
        # -- mamba
        attn_dropout, #
    ):
        '''

        Parameters
        ----------
        d_model
        num_of_heads
        gat_dropout
        attn_dropout
        add_layer_norm
        add_batch_norm
        '''
        super(GMLayer, self).__init__()

        if gat_avg_last:
            assert num_out_features == num_in_features, "[GMLayer Last AVG] num_out_features == num_in_features"
        else:
            assert num_in_features == (num_of_heads * num_out_features), "[GMLayer] num_in_features != (num_of_heads * num_out_features)"

        self.num_in_features = num_in_features
        self.num_of_heads = num_of_heads
        self.gat_dropout = gat_dropout
        self.attn_dropout = attn_dropout
        self.add_layer_norm = add_layer_norm
        self.add_batch_norm = add_batch_norm

        # -- local/global model
        self.local_model = GATLayerImp3(
            num_in_features = num_in_features,
            num_out_features = num_out_features,
            num_of_heads = num_of_heads,
            concat= (not gat_avg_last),  # last GAT layer does mean avg, the others do concat
            activation= nn.ELU() if (not gat_avg_last) else None,  # last layer just outputs raw scores
            dropout_prob=gat_dropout,
            add_skip_connection= gat_add_skip_connection,
            bias = gat_bias,
            load_trans_prob= gat_load_trans_prob,
        )

        self.global_model = Mamba(
            d_model=num_in_features,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor SSM状态维度
            d_conv=4,  # Local convolution width Local卷积核大小
            expand=1,  # Block expansion factor
        ) # 可以接收 (N, d_model)的单batch输入

        # -- normal
        if self.add_layer_norm and self.add_batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        if self.add_layer_norm: # 对于非Batch，我们per node作layer norm
            self.norm_local = nn.LayerNorm(num_in_features)
            self.norm_global = nn.LayerNorm(num_in_features)
        if self.add_batch_norm:
            self.norm_local = nn.BatchNorm1d(num_in_features)
            self.norm_global = nn.BatchNorm1d(num_in_features)
        self.dropout_local = nn.Dropout(gat_dropout)
        self.dropout_attn = nn.Dropout(attn_dropout)


        # -- ff + norm
        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(num_in_features, num_in_features * 2)
        self.ff_linear2 = nn.Linear(num_in_features * 2, num_in_features)
        if self.add_layer_norm:
            self.norm2 = nn.LayerNorm(num_in_features)
            # self.norm2 = pygnn.norm.LayerNorm(d_hidden)
            # self.norm2 = pygnn.norm.GraphNorm(num_in_features)
            # self.norm2 = pygnn.norm.InstanceNorm(d_hidden)
        if self.add_batch_norm:
            self.norm2 = nn.BatchNorm1d(num_in_features)
        self.ff_dropout1 = nn.Dropout(attn_dropout)
        self.ff_dropout2 = nn.Dropout(attn_dropout)

        # -- gated fusion
        self.W_local = nn.Linear(num_in_features, num_in_features, bias=False)
        self.W_global = nn.Linear(num_in_features, num_in_features, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        '''

        Parameters
        ----------
        data
            in_node_features (N, num_in_features)
            edge_index (2, E)
            edge_prob (E,)

        Returns
        -------

        '''
        in_nodes_features, edge_index, edge_prob = data  # unpack dataset edge_prob=(E, 1)
        h_in1 = in_nodes_features # for first residual connection, (N, num_in_features)

        # -- local gat
        h_local,_,_ = self.local_model(data) # (N, num_in_features)
        # gat 这里不需要作残差，在gat里面以及做过一次了
        # h_local = h_in1 + h_local  # Residual connection.
        if self.add_layer_norm:
            h_local = self.norm_local(h_local)
        if self.add_batch_norm:
            h_local = self.norm_global(h_local)

        # --  global mamba
        #
        deg = degree(edge_index[0], in_nodes_features.size(0)).to(torch.float32)
        deg_noise = torch.rand_like(deg).to(deg.device)  # rand_like生成值  shape(N_node) :float(0,1)
        h_ind_perm = lexsort_singlegraph(deg+deg_noise)  # 按degree展平，得到输出重排索引
        h_ind_perm_reverse = torch.argsort(h_ind_perm)
        # h_attn = self.global_model(in_nodes_features[h_ind_perm])[h_ind_perm_reverse]
        in_nodes_features = in_nodes_features[h_ind_perm].unsqueeze(0) # (1, N, d_model)
        out_nodes_features = self.global_model(in_nodes_features).squeeze(0) # (N, d_model)
        h_attn = out_nodes_features[h_ind_perm_reverse]

        h_attn = self.dropout_attn(h_attn) # dropout
        h_attn = h_in1 + h_attn # residual connection
        if self.add_layer_norm: # norm
            h_attn = self.norm_global(h_attn)
        if self.add_batch_norm:
            h_attn = self.norm_global(h_attn)

        # -- gated fusion
        # h = h_local + h_attn # (N, num_in_features)
        # h_local, h_global shape: (N, num_in_features)
        gate = self.sigmoid(self.W_local(h_local) + self.W_global(h_attn))
        h = gate * h_local + (1 - gate) * h_attn

        # FF
        h = h + self._ff_block(h)
        if self.add_layer_norm:
            h = self.norm2(h)
        if self.add_batch_norm:
            h = self.norm2(h)
        return (h, edge_index, edge_prob)

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))




if __name__ == "__main__":
    keys = torch.tensor([
        [2,5,1], [0,1,1]
    ])
    sort_index = lexsort_with_graphbatch(keys)
    print()






