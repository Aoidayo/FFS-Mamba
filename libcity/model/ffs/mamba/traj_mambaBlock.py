# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from .selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from libcity.model.bertlm.drop_path import DropPath

class TrajMamba(nn.Module):
    def __init__(
        self,
        d_model, # 模型的隐藏层维度 D
        d_state=16, # 状态空间的维度 N
        d_conv=4, # 1D卷积的卷积核大小
        expand=2, # 扩展因子 E (the controllable expansion factor)
        dt_rank="auto", # 定义输入依赖的参数Δ的秩，'auto'表示自动设置
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True, # 卷积层是否使用偏置项
        bias=False, # 其他层（如线性层）是否使用偏置项
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        aux_feature_size=0,
    ):
        '''
        Overview
        ------------
          input (B,L,D)
             │
          in_proj → [x, z] (B, L, 2*d_inner)
             │
           x → conv1d → x' (因果局部)
             │
           x' or aux → x_proj → [dt_low, B, C]
             │
           dt_low → dt_proj → Δ (d_inner)
             │
           SSM: h_t = Ā h_{t-1} + B̄ x_t
                 y_t = C_t h_t + D x_t
             │
           y_t *= silu(z)
             │
         out_proj → output (B, L, D)

        Parameters
        ----------
        d_model  /D 轨迹嵌入维度                  128
        d_state SSM状态空间维度               16
        d_conv  causal_conv1d卷积核大小      4
        expand  内部维度扩展因子 E
        -- dt's
        dt_rank     时间步 Δ 的低秩秩,  int / "auto", auto -> D/16
        dt_min      Δ 的取值范围（softplus 后）, 0.001~0.1
        dt_max
        dt_init     Δ 投影权重初始化方式
        dt_scale
        dt_init_floor
        conv_bias   卷积是否加偏置
        bias        线性层是否添加偏置
        use_fast_path   是否使用融合算子（加速）
        layer_idx       当前层编号（用于缓存）
        device
        dtype
        aux_feature_size    辅助特征维度（如速度、航向）， default=3
        '''


        # -- parameter
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        # Mamba 发现：在 SSM 中扩展隐藏维度能显著提升性能，类似 Transformer 的 FFN（4×）。
        self.d_inner = int(self.expand * self.d_model) # expanding the model dimension D by the controllable expansion factor E
        # Δ 是时间步参数，但需依赖输入 → 用 低秩投影 减少参数量：
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.aux_feature_size = aux_feature_size


        # -- model
        # -- model -- Linear(d_model, d_inner*2)
        # 输出的(B,T,d_inner*2) 将会被切分为 x,z 即(B, T, d_inner)
        '''
        xz = in_proj(hidden_states)
        x, z = xz.chunk(2, dim=1)
        ------
        x：进入 SSM 主路径
        z：门控分支（类似 GRU 的 z 门）
        '''
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # -- model -- conv1d
        # 捕捉 短时局部动态
        # B*in_channels*L → B*out_channels*(L + d_conv-1)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv, # 局部感受野
            groups=self.d_inner, # 深度可分离卷积（Depthwise） → 每通道独立卷积，参数少
            padding=d_conv - 1, # 保证输出长度 = 输入长度
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        # -- model -- 参数投影，生成Δ B C
        # 输入，输出，以及时间步状态依赖 输入变化
        self.x_proj = nn.Linear(self.aux_feature_size if self.aux_feature_size else self.d_inner,
                                self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        # -- -- Δ
        # 低秩 Δ → 高维 Δ，(dt_rank) -> (d_inner)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        # Δproj 权重初始化（保持方差）
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # dt_proj.bias 初始化 → 控制 Δ 范围，保证 Δ 在合理范围内（避免梯度爆炸/消失）
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # SSM 的对角矩阵 A（可学习）
        ## ssm参数 A、D 与输入无关
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", # h,n = d_state
            d=self.d_inner,
        ).contiguous() # (d_inner, d_state)
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True # 告诉scheduler, A 不参与 L2 正则（经验最佳）

        # D "skip connection" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    # Input: x(𝙱, 𝙻, 𝙳) → Output: y(𝙱, 𝙻, 𝙳)
    def forward(self, hidden_states, aux_features=None, inference_params=None):
        """
        Params
        ---------
        hidden_states: (B, L, D), D=d_model
        aux_features: (B, L, 3), aux_features
        inference_params: Default To None

        Returns
        ---------
        same shape as hidden_states
        traj_h: (B, L, D)
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, aux_features, conv_state, ssm_state)
                return out

        # 将输入映射到 xz
        # We do matmul and transpose BLH -> HBL at the same time
        # in_proj (d_inner*2, d_model) @ (d_model, batch_size, seq_len ) = (d_inner*2, batch_size*seq_len)
        # rearrange : (batch_size, d_inner*2, seq_len)
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", # shape [d_inner * 2, (B L)] -> (B, d_inner * 2, L)
            l=seqlen,
        ) 
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        # discrete BCΔ
        src_params = None
        if self.aux_feature_size:
            src_params = rearrange(aux_features, "b l d -> b d l")

        # A
        A = -torch.exp(self.A_log.float())  # shape (d_inner, d_state)
        
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                src_params=src_params,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1) 
            
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen]) # shape (B, d_inner, L)
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            
            x_dbl = self.x_proj(rearrange(src_params if self.aux_feature_size else x, "b d l -> (b l) d"))  # (bl d)
            
            # Δ: [(B L), dt_rank]   B, C: [(B L), d_state]
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            dt = self.dt_proj.weight @ dt.t() # shape [d_inner, (B L)]
            
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen) # shape (B, d_inner, L)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # shape (B, d_state, L)
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # shape (B, d_state, L)
            
            assert self.activation in ["silu", "swish"]
            
            y = selective_scan_fn(
                x, # (B, d_inner, L)
                dt, # (B, d_inner, L)
                A, # (d_inner, d_state)
                B, # (B, d_state, L)
                C, # (B, d_state, L)
                self.D.float(), # shape (d_inner)
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ) #  shape (B, d_inner, L)
            
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y) # (B, L, d_inner) -> (B, L, D)

        return out

    def step(self, hidden_states, aux_features, conv_state, ssm_state): # hidden_states: (B 1 d_model)
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        if self.aux_feature_size:
            assert aux_features.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)   D: d_inner, the expanded model dimension
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(aux_features.squeeze(1) if self.aux_feature_size else x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A)) # (B d_inner d_state)   b:B, d:d_inner(D), n:d_state
            dB = torch.einsum("bd,bn->bdn", dt, B) # (B d_inner d_state)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB) # (B d_inner d_state)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C) # (B d_inner)
            y = y + self.D.to(dtype) * x # (B d_inner)   self.D:(D), x:(B D)
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y) # (B d_model)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self,
        d_model,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=True,
        residual_in_fp32=False,
        drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(d_model) # 预定义参数，只需要指定d_model的TrajMamba可调用函数/类
        self.norm = norm_cls(d_model) # RMS/Layer Norm
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm: #
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"



    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, aux_features=None, inference_params=None
    ):
        r"""将输入送入Encoder层

        Args:
            hidden_states: encoder层输入的序列 (required). shape (B, L, D)
                当前层输入，形状 (B, L, D)
            residual: hidden_states = Mixer(LN(residual)). shape (B, L, D)
                残差支路的输入（来自上一层），也可为 None
            aux_features: 用于离散化mamba初始参数，可以为None
        
        Returns: 
            hidden_states, residual: updated params with same shape
                residual = hidden_states (+ residual)
                hidden_states = Mixer(LN(residual))
        """
        # 非 fused 残差+正则
        if not self.fused_add_norm:
            # -- Pre(L)N: 先残差 后Norma
            # 第一层：residual 初始为None，不需要skip-connection
            # 第i层：实施skip-connection
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states # add
            # 正则
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype)) # LayerNorm
            if self.residual_in_fp32: # fp32保存残差
                residual = residual.to(torch.float32)
        # 默认 使用fused kernel加速
        else:
            # 1. add residual 2.layernorm 3.输出hidden,residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                x = self.drop_path(hidden_states),
                weight = self.norm.weight,
                bias = self.norm.bias,
                residual=residual,
                prenorm=True, # need to return the residual
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # mamba输出
        hidden_states = self.mixer(hidden_states, aux_features=aux_features, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)