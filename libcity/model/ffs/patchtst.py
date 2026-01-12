'''
57.7
'''
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# 你已有的 RoadGM / RoPE embedding
from libcity.model.ffs.road_gm import RoadGM
from libcity.model.bertlm.positional_embedding import RotaryPositionalEmbedding


# =========================
# Config (变长，不再需要 pad_length)
# =========================
@dataclass
class PatchTSTRoadConfig:
    vocab_size: int = 6253
    min_seq_len: int = 11

    # PatchTST
    d_model: int = 256
    patch_length: int = 4
    patch_stride: int = 2
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    ffn_ratio: int = 4
    pre_norm: bool = True

    # ---- RoadView embedding 配置（对齐你给的 BERTEmbedding）----
    add_time_in_day: bool = True
    add_day_in_week: bool = True
    add_pe: bool = True
    add_gat: bool = True

    node_fea_dim: int = 10

    roadgm_d_model: int = 256
    roadgm_add_layer_norm: bool = True
    roadgm_add_batch_norm: bool = False
    roadgm_gat_num_heads_per_layer: tuple = (8, 16, 1)
    roadgm_gat_num_features_per_layer: tuple = (16, 16, 256)
    roadgm_gat_bias: bool = True
    roadgm_gat_dropout: float = 0.6
    roadgm_gat_avg_last: bool = True
    roadgm_gat_load_trans_prob: bool = True
    roadgm_gat_add_skip_connection: bool = True
    roadgm_mamba_attn_dropout: float = 0.1


# =========================
# Road Embedding (对齐你给的 BERTEmbedding)
# =========================
class BERTEmbedding(nn.Module):
    """
    输入 sequence: (B, T, F)
      - sequence[:,:,0] = road_id (Long)
      - sequence[:,:,2] = mins (0~1440)  (如果 add_time_in_day=True)
      - sequence[:,:,3] = weekday (1~7)  (如果 add_day_in_week=True)
    graph_dict:
      node_features: (vocab_size, node_fea_dim)
      edge_index: (2, E)
      edge_index_trans_prob: (E, 1) or (E,)
    输出: (B, T, d_model)
    """
    def __init__(self, cfg: PatchTSTRoadConfig):
        super().__init__()
        self.add_time_in_day = cfg.add_time_in_day
        self.add_day_in_week = cfg.add_day_in_week
        self.add_pe = cfg.add_pe
        self.add_gat = cfg.add_gat
        self.d_model = cfg.d_model

        if self.add_gat:
            self.token_embedding = RoadGM(
                in_feature=cfg.node_fea_dim,
                d_model=cfg.roadgm_d_model,
                add_layer_norm=cfg.roadgm_add_layer_norm,
                add_batch_norm=cfg.roadgm_add_batch_norm,
                # gat
                num_features_per_layer=list(cfg.roadgm_gat_num_features_per_layer),
                num_heads_per_layer=list(cfg.roadgm_gat_num_heads_per_layer),
                gat_bias=cfg.roadgm_gat_bias,
                gat_dropout=cfg.roadgm_gat_dropout,
                gat_avg_last=cfg.roadgm_gat_avg_last,
                gat_load_trans_prob=cfg.roadgm_gat_load_trans_prob,
                gat_add_skip_connection=cfg.roadgm_gat_add_skip_connection,
                # mamba
                attn_dropout=cfg.roadgm_mamba_attn_dropout,
            )
        else:
            # 若不用 RoadGM，你也可以在这里用 nn.Embedding 作为替代
            self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if self.add_pe:
            self.position_embedding = RotaryPositionalEmbedding(d_model=cfg.d_model)

        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, cfg.d_model, padding_idx=0)

        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, cfg.d_model, padding_idx=0)

        self.dropout = nn.Dropout(p=cfg.dropout)

        # 若 RoadGM 输出维度不是 d_model，做个投影（保险）
        self.out_proj = None
        if cfg.add_gat and cfg.roadgm_d_model != cfg.d_model:
            self.out_proj = nn.Linear(cfg.roadgm_d_model, cfg.d_model)

    def forward(self, sequence: torch.Tensor, position_ids=None, graph_dict=None) -> torch.Tensor:
        if self.add_gat:
            if graph_dict is None:
                raise ValueError("graph_dict is required when add_gat=True.")
            x = self.token_embedding(
                node_features=graph_dict["node_features"],
                edge_index_input=graph_dict["edge_index"],
                edge_prob_input=graph_dict["edge_index_trans_prob"],
                x=sequence[:, :, 0].long(),
            )  # (B,T,roadgm_d_model)
            if self.out_proj is not None:
                x = self.out_proj(x)  # (B,T,d_model)
        else:
            x = self.token_embedding(sequence[:, :, 0].long())  # (B,T,d_model)

        if self.add_pe:
            # 与你原实现保持一致：直接 x += RoPE(x)
            x = x + self.position_embedding(x, position_ids)

        if self.add_time_in_day:
            x = x + self.daytime_embedding(sequence[:, :, 2].long())

        if self.add_day_in_week:
            x = x + self.weekday_embedding(sequence[:, :, 3].long())

        return self.dropout(x)


# =========================
# Patch-level SinCos PE
# =========================
def sincos_pe(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(length, dim, device=device)
    pos = torch.arange(0, length, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# =========================
# Patchify (动态 T)
# =========================
class Patchify(nn.Module):
    """
    token_emb (B,T,D) -> patches (B,Np,Pl,D)
    Np = floor((T-Pl)/S)+1
    """
    def __init__(self, patch_length: int, patch_stride: int):
        super().__init__()
        self.Pl = patch_length
        self.S = patch_stride

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        B, T, D = token_emb.shape
        if T < self.Pl:
            raise ValueError(f"Sequence length T={T} must be >= patch_length={self.Pl}.")
        patches = token_emb.unfold(dimension=1, size=self.Pl, step=self.S).permute(0, 1, 3, 2).contiguous()
        return patches  # (B,Np,Pl,D)


def patches_to_tokens_mean_overlap(
    patch_repr: torch.Tensor,         # (B,Np,D)
    padding_masks: torch.Tensor,      # (B,T) bool/0-1
    patch_length: int,
    patch_stride: int,
    T: int,
) -> torch.Tensor:
    device = patch_repr.device
    B, Np, D = patch_repr.shape

    starts = torch.arange(Np, device=device) * patch_stride
    offsets = torch.arange(patch_length, device=device)
    idx = (starts[:, None] + offsets[None, :]).reshape(1, -1).expand(B, -1)  # (B, Np*Pl)

    padding_masks = padding_masks.bool()
    mask_flat = padding_masks.gather(1, idx).float()  # (B, N)

    src = patch_repr[:, :, None, :].expand(B, Np, patch_length, D).reshape(B, -1, D)
    src = src * mask_flat.unsqueeze(-1)

    out = torch.zeros(B, T, D, device=device)
    cnt = torch.zeros(B, T, 1, device=device)

    out.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, D), src)
    cnt.scatter_add_(1, idx.unsqueeze(-1), mask_flat.unsqueeze(-1))

    out = out / cnt.clamp_min(1.0)
    out = out * padding_masks.unsqueeze(-1)
    return out


# =========================
# PatchTST Encoder Layer (纯 PyTorch)
# =========================
class PatchTSTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ffn_ratio: int, pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = d_model * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            h = self.ln1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.drop1(attn_out)

            h = self.ln2(x)
            x = x + self.drop2(self.ffn(h))
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.ln1(x + self.drop1(attn_out))
            x = self.ln2(x + self.drop2(self.ffn(x)))
        return x


class PatchTSTEncoder(nn.Module):
    """
    token_emb (B,T,D) -> patchify (B,Np,Pl,D) -> mean pool -> (B,Np,D)
    patch tokens 上做 Transformer 编码。
    """
    def __init__(self, cfg: PatchTSTRoadConfig):
        super().__init__()
        self.cfg = cfg
        self.patchify = Patchify(cfg.patch_length, cfg.patch_stride)
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList([
            PatchTSTEncoderLayer(
                d_model=cfg.d_model,
                n_heads=cfg.num_heads,
                dropout=cfg.dropout,
                ffn_ratio=cfg.ffn_ratio,
                pre_norm=cfg.pre_norm
            )
            for _ in range(cfg.num_layers)
        ])

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(token_emb)     # (B,Np,Pl,D)
        patch_tokens = patches.mean(dim=2)     # (B,Np,D)

        Np = patch_tokens.size(1)
        pe = sincos_pe(Np, self.cfg.d_model, patch_tokens.device)  # (Np,D)
        x = self.pos_drop(patch_tokens + pe.unsqueeze(0))

        for layer in self.layers:
            x = layer(x)
        return x  # (B,Np,D)


# =========================
# RoadView-style PatchTST Wrapper (用你的 BERTEmbedding)
# =========================
class PatchTSTRoadView(nn.Module):
    """
    forward 返回与 RoadView 对齐：
      road_embedding_output: (B,T,256)
      mean_pooled: (B,256)
      logits: (B,T,vocab_size)
    """
    def __init__(self, cfg: PatchTSTRoadConfig):
        super().__init__()
        self.cfg = cfg

        # 直接用你给的 embedding 逻辑
        self.embedding = BERTEmbedding(cfg)

        self.encoder = PatchTSTEncoder(cfg)
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(
        self,
        x: torch.Tensor,                 # (B,T,F) 你的 sequence
        padding_masks: torch.Tensor,     # (B,T) bool/0-1 True=有效
        batch_temporal_mat=None,
        graph_dict=None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        B, T = x.shape[0], x.shape[1]

        padding_masks = padding_masks.bool()
        valid_lengths = padding_masks.sum(dim=1).long()
        if (valid_lengths < self.cfg.min_seq_len).any():
            raise ValueError(f"Found sequence with valid_length < {self.cfg.min_seq_len}.")
        if T < self.cfg.patch_length:
            raise ValueError(f"T={T} < patch_length={self.cfg.patch_length}.")

        # (B,T,D) embedding
        token_emb = self.embedding(sequence=x, position_ids=None, graph_dict=graph_dict)
        token_emb = token_emb * padding_masks.unsqueeze(-1)

        # PatchTST over patches
        patch_repr = self.encoder(token_emb)  # (B,Np,D)

        # 回填到 token 级，便于对齐 RoadView 输出/做 MLM
        road_embedding_output = patches_to_tokens_mean_overlap(
            patch_repr=patch_repr,
            padding_masks=padding_masks,
            patch_length=self.cfg.patch_length,
            patch_stride=self.cfg.patch_stride,
            T=T,
        )  # (B,T,D)

        # mean pool (按 padding_masks)
        mask = padding_masks.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_pooled = (road_embedding_output * mask).sum(dim=1) / denom  # (B,D)

        logits = self.mlm_head(road_embedding_output)  # (B,T,vocab)

        return road_embedding_output, mean_pooled, logits


# -------------------------
# quick sanity check
# -------------------------
if __name__ == "__main__":
    cfg = PatchTSTRoadConfig(
        vocab_size=6253,
        min_seq_len=11,
        d_model=256,
        patch_length=4,
        patch_stride=2,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        add_gat=True,
        add_pe=True,
        add_time_in_day=True,
        add_day_in_week=True,
        node_fea_dim=10,
        roadgm_d_model=256,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchTSTRoadView(cfg).to(device)

    # batch 内变长 pad 到 max_len
    B = 4
    lengths = torch.tensor([11, 23, 64, 80], device=device)
    T = int(lengths.max().item())

    # sequence: (B,T,F) -> 这里至少要有 col0(road_id), col2(mins), col3(weekday)
    Fdim = 5
    seq = torch.zeros(B, T, Fdim, device=device)
    seq[:, :, 0] = torch.randint(0, cfg.vocab_size, (B, T), device=device)  # road_id
    seq[:, :, 2] = torch.randint(0, 1441, (B, T), device=device)            # mins
    seq[:, :, 3] = torch.randint(1, 8, (B, T), device=device)               # weekday

    padding_masks = (torch.arange(T, device=device)[None, :] < lengths[:, None])  # True=有效

    # graph_dict 必须提供给 RoadGM
    # 注意：这里是 dummy 结构，实际用你的 road graph 填
    E = 100
    graph_dict = {
        "node_features": torch.randn(cfg.vocab_size, cfg.node_fea_dim, device=device),
        "edge_index": torch.randint(0, cfg.vocab_size, (2, E), device=device),
        "edge_index_trans_prob": torch.rand(E, 1, device=device),
    }

    y_seq, y_pool, y_logits = model(seq, padding_masks, graph_dict=graph_dict)
    print(y_seq.shape, y_pool.shape, y_logits.shape)
    # (B,T,256) (B,256) (B,T,vocab_size)
