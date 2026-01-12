import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch import Tensor
import torch_geometric
# from GCNIIlayer import GCNIIdenseConv
from torch_geometric.nn import GATConv, GCNConv, ChebConv, GCN2Conv, SGConv
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn import TransformerConv
import scipy.sparse as sp
from scipy.special import comb
from torch_sparse import SparseTensor
from torch_geometric.typing import (
    Adj,
    OptTensor,
)
from torch.nn import Linear, ModuleList, Module, Dropout, ReLU, GELU, Sequential

from einops import rearrange, repeat, einsum

from mamba_ssm import Mamba2


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GCNII(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCNII, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(dataset.num_features, args.hidden))
        # for _ in range(args.layer_num):
        #     self.convs.append(GCNIIdenseConv(args.hidden, args.hidden))
        self.dropout = args.dropout
        self.lamda = args.lamda
        self.alpha = args.alpha
        self.convs.append(torch.nn.Linear(args.hidden, dataset.num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i, con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout, training=self.training)
            beta = math.log(self.lamda / (i + 1) + 1)
            x = F.relu(con(x, edge_index, self.alpha, _hidden[0], beta, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_mamba_liner(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=False):
        super(GCN_mamba_liner, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = input @ self.weight
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN_mamba_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_mamba_Net, self).__init__()
        self.dropout = args.mamba_dropout
        self.args = args
        self.a = self.args.alpha
        self.b = self.args.graph_weight
        self.lin1 = GCN_mamba_liner(dataset.num_features, args.d_model, with_bias=args.bias)
        self.LayerNorm_1 = torch.nn.LayerNorm(self.args.d_model, eps=1e-12)
        self.mamba_global_attention = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=args.d_model,
            d_state=8,
            headdim=8,
            d_conv=4,
            expand=1,
        )

        self.LayerNorm_2 = torch.nn.LayerNorm(self.args.d_model, eps=1e-12)
        self.bn_1 = torch.nn.BatchNorm1d(args.d_model)
        self.layer_num = args.layer_num
        self.mamba = GCN_mamba_block(args, dataset)
        self.norm_1 = RMSNorm(args.d_model)
        self.bn_2 = torch.nn.BatchNorm1d(args.d_model)
        self.lin2 = GCN_mamba_liner(args.d_model, dataset.num_classes, with_bias=args.bias)
        self.bn_3 = torch.nn.BatchNorm1d(args.d_model)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.mamba.reset_parameters()

    def forward(self, x, adj):
        x_input = x
        adj_t = adj

        x_input = self.lin1(x_input)

        global_attention_input = x_input.reshape(1, x_input.shape[0], x_input.shape[1])

        global_attention_input = F.dropout(global_attention_input, p=self.dropout, training=self.training)

        global_attention_input_flip = torch.flip(global_attention_input, dims=[1])

        global_attention_output = self.mamba_global_attention(global_attention_input)
        global_attention_output_flip = self.mamba_global_attention(global_attention_input_flip)

        global_attention_output = global_attention_output + torch.flip(global_attention_output_flip, dims=[1])

        global_attention_output = F.dropout(global_attention_output, p=self.dropout, training=self.training)
        global_attention_output = F.relu(global_attention_output)
        global_attention_output = torch.squeeze(global_attention_output) * (1 - self.a) + self.a * x_input

        x = self.bn_1(x_input)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        all_layers_output = self.mamba(x, adj_t, self.args.layer_num)
        output = (all_layers_output[:, -1, :] + x_input) * self.b + global_attention_output * (1 - self.b)
        output = self.bn_2(output)
        output = F.relu(output)
        all_layers_output = F.relu(all_layers_output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        all_layers_output = F.dropout(all_layers_output, p=self.dropout, training=self.training)
        y = self.lin2(output)

        return all_layers_output, F.log_softmax(y, dim=-1)


class GCN_mamba_block(torch.nn.Module):
    def __init__(self, args, dataset):
        super(GCN_mamba_block, self).__init__()
        self.args = args

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        if args.dataset in ['ogbn-arxiv']:
            self.conv1 = GCNConv(args.d_model, args.d_model)
            self.conv2 = GCNConv(args.d_model, args.d_model)
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(args.d_model))
            self.conv = []
            self.hidden_layer_num = 1
            for i in range(self.hidden_layer_num):
                self.conv.append(GCNConv(args.d_model, args.d_model).to(args.device))
                self.bns.append(torch.nn.BatchNorm1d(args.d_model))
            self.bns.append(torch.nn.BatchNorm1d(args.d_model))
        self.x_proj = GCN_mamba_liner(args.d_inner, args.dt_rank + args.d_state * 2, with_bias=args.bias)
        self.dropout = args.mamba_dropout
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = GCN_mamba_liner(args.dt_rank, args.d_inner, with_bias=args.bias)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)

        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = GCN_mamba_liner(args.d_inner, args.d_model, with_bias=args.bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.x_proj.reset_parameters()
        self.dt_proj.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, x, adj, layer_num):
        """
        Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """
        alpha = 0.05
        features = [x]
        xi = x
        for i in range(layer_num - 1):
            xi = adj @ xi
            xi = (1 - alpha) * xi + alpha * x
            features.append(xi)

        x = torch.stack(features, dim=0).transpose(0, 1)
        y = self.ssm(x, self.args, adj)
        output = self.out_proj(y)

        return output

    def ssm(self, x, args, adj):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        x_dbl = F.relu(x_dbl)
        x_dbl = F.dropout(x_dbl, p=self.dropout, training=self.training)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D, args, adj)

        return y

    def selective_scan(self, u, delta, A, B, C, D, args, adj):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        手写版的 Mamba SSM 递推（序列扫描）。
        Args:
            A 是固定衰减参数（learned）
            B(t)、C(t)、Δ(t) 由输入 x 生成（input-dependent）
            u(t)：Mamba 的输入序列， u[:,i] 第i阶邻域特征
            x(t)：Mamba 的隐藏状态
            t: 原版Mamba中表示时间、现表示为图的k-hop邻域阶数


        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """

        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(args.layer_num):
            x = deltaA[:, i] * x + deltaB_u[:, i]

            x = F.dropout(x, p=self.dropout, training=self.training)

            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')

            ys.append(y)

        y = torch.stack(ys, dim=1)

        y = y + u * D

        return y


class RMSNorm(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output