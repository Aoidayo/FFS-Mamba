import os
import json
import datetime
from logging import getLogger
from typing import Dict, List, Optional

import numpy as np
import torch
import pandas as pd

from libcity.evaluator.bertlm.abstract_evaluator import AbstractEvaluator
from libcity.utils import ensure_dir


# =========================
# Vectorized top-k stats
# =========================
@torch.no_grad()
def top_k_stats(loc_pred, loc_true, topk: int):
    """
    loc_pred: (N, C) logits/prob
    loc_true: (N,) int
    Returns:
        hit: int
        mrr_sum: float   (sum over samples of 1/rank if hit else 0)
        dcg_sum: float   (sum over samples of 1/log2(rank+1) if hit else 0)  rank starts from 1
        topk_idx: (N, topk) np.int64
        hit_mask: (N,) torch.bool
        rank_pos: (N,) torch.long, 0-based position within topk; for non-hit = topk
    """
    pred = torch.as_tensor(loc_pred, dtype=torch.float32, device="cpu")
    true = torch.as_tensor(loc_true, dtype=torch.long, device="cpu")

    _, idx = torch.topk(pred, k=topk, dim=1)  # (N, k)
    # match matrix
    matches = idx.eq(true[:, None])  # (N, k)

    # find first match position per row
    N = idx.size(0)
    pos = torch.arange(topk, device=idx.device).view(1, topk).expand(N, topk)  # (N,k)
    pos_masked = pos.masked_fill(~matches, topk)  # non-match -> topk
    rank_pos, _ = pos_masked.min(dim=1)  # (N,) 0..k-1 or k
    hit_mask = rank_pos < topk

    hit = int(hit_mask.sum().item())

    if hit > 0:
        rank = (rank_pos[hit_mask] + 1).to(torch.float32)  # 1..k
        mrr_sum = float((1.0 / rank).sum().item())
        # dcg: 1/log2(rank+1), because original used log2(rank_index+2) where rank_index=rank-1
        dcg_sum = float((1.0 / torch.log2(rank + 1.0)).sum().item())
    else:
        mrr_sum = 0.0
        dcg_sum = 0.0

    return hit, mrr_sum, dcg_sum, idx.numpy().astype(np.int64), hit_mask, rank_pos


def _safe_div(a: np.ndarray, b: np.ndarray, zero_division: float = 0.0) -> np.ndarray:
    out = np.full_like(a, fill_value=zero_division, dtype=np.float64)
    mask = b > 0
    out[mask] = a[mask] / b[mask]
    return out


# =========================
# Evaluator
# =========================
class FFS_Classify_Evaluator(AbstractEvaluator):
    """
    单标签多分类 + Top-k ranking 指标
    """

    def __init__(self, config):
        self.config = config
        self._logger = getLogger()

        # 你希望保存到 csv 的列名（每个都会输出 @k）
        self.metrics = config.get(
            "metrics",
            [
                "Accuracy",
                "MacroPrecision", "MacroRecall", "MacroF1",
                "MicroPrecision", "MicroRecall", "MicroF1",
                "MRR", "MAP", "NDCG",
            ],
        )
        self.allowed_metrics = [
            "Accuracy",
            "MacroPrecision", "MacroRecall", "MacroF1",
            "MicroPrecision", "MicroRecall", "MicroF1",
            "MRR", "MAP", "NDCG",
        ]

        self.save_modes = config.get("save_modes", ["csv", "json"])
        self.topk = config.get("topk", [1, 5, 10])

        self.clear()
        self._check_config()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError("Evaluator metrics must be a list")
        for m in self.metrics:
            if m not in self.allowed_metrics:
                raise ValueError(f"Metric '{m}' not in allowed_metrics")

    def clear(self):
        self.result: Dict[str, float] = {}

        self.intermediate_result: Dict[str, float] = {}
        self.intermediate_result["total"] = 0

        # ranking-style accumulators
        for k in self.topk:
            self.intermediate_result[f"hit@{k}"] = 0
            self.intermediate_result[f"mrr_sum@{k}"] = 0.0
            self.intermediate_result[f"dcg_sum@{k}"] = 0.0

        # classification accumulators (macro/micro)
        self._num_classes: Optional[int] = None
        self._support: Optional[np.ndarray] = None  # (C,)
        self._tp_at_k: Dict[int, Optional[np.ndarray]] = {k: None for k in self.topk}       # (C,)
        self._pred_cnt_at_k: Dict[int, Optional[np.ndarray]] = {k: None for k in self.topk} # (C,)

    def _ensure_class_buffers(self, num_classes: int):
        if self._num_classes is not None:
            return
        self._num_classes = int(num_classes)
        C = self._num_classes
        self._support = np.zeros(C, dtype=np.int64)
        for k in self.topk:
            self._tp_at_k[k] = np.zeros(C, dtype=np.int64)
            self._pred_cnt_at_k[k] = np.zeros(C, dtype=np.int64)

    def collect(self, batch):
        """
        batch:
            'loc_true': (N,)
            'loc_pred': (N, C)
        """
        if not isinstance(batch, dict):
            raise TypeError("evaluator.collect input is not a dict")

        y_true = batch["loc_true"]
        y_pred = batch["loc_pred"]

        # infer shapes / num classes
        pred_t = torch.as_tensor(y_pred)
        if pred_t.dim() != 2:
            raise ValueError(f"loc_pred must be (N,C), got {tuple(pred_t.shape)}")
        N, C = int(pred_t.shape[0]), int(pred_t.shape[1])

        self._ensure_class_buffers(C)

        true = torch.as_tensor(y_true, dtype=torch.long, device="cpu")
        self.intermediate_result["total"] += int(true.numel())

        # support per class
        support = np.bincount(true.numpy(), minlength=C).astype(np.int64)
        self._support += support

        # per-k
        for k in self.topk:
            hit, mrr_sum, dcg_sum, topk_idx, hit_mask, _ = top_k_stats(y_pred, y_true, k)

            self.intermediate_result[f"hit@{k}"] += hit
            self.intermediate_result[f"mrr_sum@{k}"] += mrr_sum
            self.intermediate_result[f"dcg_sum@{k}"] += dcg_sum

            # pred count per class in top-k list
            pred_cnt = np.bincount(topk_idx.reshape(-1), minlength=C).astype(np.int64)
            self._pred_cnt_at_k[k] += pred_cnt

            # TP per class: those samples whose true label is hit within top-k
            if hit > 0:
                tp = np.bincount(true.numpy()[hit_mask.numpy()], minlength=C).astype(np.int64)
                self._tp_at_k[k] += tp

    def evaluate(self):
        total = int(self.intermediate_result["total"])
        if total == 0:
            # fill zeros
            for k in self.topk:
                for m in self.metrics:
                    self.result[f"{m}@{k}"] = 0.0
            return self.result

        C = int(self._num_classes)
        support = self._support.astype(np.int64)

        valid = support > 0  # macro only over seen classes

        for k in self.topk:
            hit = int(self.intermediate_result[f"hit@{k}"])
            mrr_sum = float(self.intermediate_result[f"mrr_sum@{k}"])
            dcg_sum = float(self.intermediate_result[f"dcg_sum@{k}"])

            # ---------- ranking / top-k acc ----------
            self.result[f"Accuracy@{k}"] = hit / total
            self.result[f"MRR@{k}"] = mrr_sum / total
            # 单正例（单标签）时：AP_i = 1/rank_i (hit else 0)，所以 MAP == MRR
            self.result[f"MAP@{k}"] = self.result[f"MRR@{k}"]
            self.result[f"NDCG@{k}"] = dcg_sum / total

            # ---------- macro/micro classification-like (Top-k hit as TP) ----------
            tp = self._tp_at_k[k].astype(np.int64)
            pred_cnt = self._pred_cnt_at_k[k].astype(np.int64)

            fp = pred_cnt - tp
            fn = support - tp

            p_c = _safe_div(tp.astype(np.float64), (tp + fp).astype(np.float64), zero_division=0.0)
            r_c = _safe_div(tp.astype(np.float64), (tp + fn).astype(np.float64), zero_division=0.0)
            f1_c = np.zeros(C, dtype=np.float64)
            denom = p_c + r_c
            mask = denom > 0
            f1_c[mask] = 2 * p_c[mask] * r_c[mask] / denom[mask]

            self.result[f"MacroPrecision@{k}"] = float(p_c[valid].mean()) if valid.any() else 0.0
            self.result[f"MacroRecall@{k}"] = float(r_c[valid].mean()) if valid.any() else 0.0
            self.result[f"MacroF1@{k}"] = float(f1_c[valid].mean()) if valid.any() else 0.0

            # micro over all classes
            TP = int(tp.sum())
            Pred = int(pred_cnt.sum())  # ~= total*k
            micro_p = (TP / Pred) if Pred > 0 else 0.0
            micro_r = TP / total
            micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

            self.result[f"MicroPrecision@{k}"] = float(micro_p)
            self.result[f"MicroRecall@{k}"] = float(micro_r)
            self.result[f"MicroF1@{k}"] = float(micro_f1)

        return self.result

    def save_result(self, save_path, filename=None):
        self.evaluate()
        ensure_dir(save_path)

        if filename is None:
            filename = (
                str(self.config["exp_id"])
                + "_"
                + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                + "_"
                + self.config["model"]
                + "_"
                + self.config["dataset"]
            )

        if "json" in self.save_modes:
            path = os.path.join(save_path, f"{filename}.json")
            with open(path, "w") as f:
                json.dump(self.result, f, indent=1)
            self._logger.info(f"Evaluate result saved at {path}")

        df = None
        if "csv" in self.save_modes:
            data = {m: [] for m in self.metrics}
            for k in self.topk:
                for m in self.metrics:
                    data[m].append(self.result.get(f"{m}@{k}", 0.0))

            df = pd.DataFrame(data, index=self.topk)

            path = os.path.join(save_path, f"{filename}.csv")
            df.to_csv(path, index=False)
            self._logger.info(f"Evaluate result saved at {path}")

            # 打印完整（避免 pandas 自动省略）
            with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", 200,
                "display.max_colwidth", None,
            ):
                self._logger.info("\n" + df.to_string())

        return df
