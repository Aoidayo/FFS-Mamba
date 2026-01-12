import os
import json
import torch
import datetime
from logging import getLogger
import pandas as pd
import numpy as np
from libcity.evaluator.bertlm.abstract_evaluator import AbstractEvaluator
# from libcity.evaluator.utils import top_k
from libcity.utils import ensure_dir


def top_k(loc_pred, loc_true, topk):
    loc_pred = torch.FloatTensor(loc_pred)  # (batch_size * output_dim)
    val, index = torch.topk(loc_pred, topk, 1)
    index = index.numpy()  # (batch_size * topk)
    hit = 0
    rank = 0.0
    dcg = 0.0
    for i, p in enumerate(index):  # i->batch, p->(topk,)
        target = loc_true[i]
        if target in p:
            hit += 1
            rank_list = list(p)
            rank_index = rank_list.index(target)
            # rank_index is start from 0, so need plus 1
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg


class FFSDP_Evaluator(AbstractEvaluator):

    def __init__(self, config):
        '''

        Args:
            config: ConfigParser
            data_feature: not in use
        '''
        # ['Precision', 'Recall', 'F1', 'MRR', 'NDCG']
        self.metrics = config.get('metrics', ["Precision", "Recall", "F1", "MRR", "MAP", "NDCG"])
        self.config = config
        # ['csv', 'json']
        self.save_modes = config.get('save_modes', ['csv', 'json'])
        self.topk = config.get('topk', [1]) # [1,5,10]
        self.allowed_metrics = ['Precision', 'Recall', 'F1', 'MRR', 'MAP', 'NDCG',
                                'Acc', 'MacroF1', 'MacroRecall', 'Acc@k'  # Acc@k 只是标识，实际会生成 Acc@{k}
                                ]
        self.clear()
        self._logger = getLogger()
        self._check_config()

    def _check_config(self):
        '''
        检查config中的metrics
            1. isinstance(self.metrics, list)
            2. metrics in allowed_metrics

        Returns:

        '''
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i not in self.allowed_metrics:
                raise ValueError('the metric is not allowed in ClassificationEvaluator')

    def collect(self, batch):
        '''
        调用topk计算hitk, rankk, dcgk

        Args:
            batch: evaluate_input (dict)
                'loc_true': (num_active, )
                'loc_pred': (num_active, n_class)

        Returns:

        '''
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        total = len(batch['loc_true'])
        self.intermediate_result['total'] += total

        # ---- update confusion matrix for MacroF1 / MacroRecall / Acc ----
        loc_pred = batch['loc_pred']
        loc_true = batch['loc_true']

        if not torch.is_tensor(loc_pred):
            loc_pred = torch.tensor(loc_pred)
        if not torch.is_tensor(loc_true):
            loc_true = torch.tensor(loc_true)

        # loc_pred: (B, C), loc_true: (B,)
        loc_pred = loc_pred.detach()
        loc_true = loc_true.detach().long().view(-1)

        if self.num_class is None:
            self.num_class = loc_pred.shape[1]
            self.cm = torch.zeros((self.num_class, self.num_class), dtype=torch.long)  # on CPU

        pred_index = loc_pred.argmax(dim=1).long().cpu()
        true_index = loc_true.cpu()

        # bincount on flattened (true, pred)
        flat = true_index * self.num_class + pred_index
        binc = torch.bincount(flat, minlength=self.num_class * self.num_class)
        self.cm += binc.view(self.num_class, self.num_class)

        for k in self.topk:
            hit, rank, dcg = top_k(batch['loc_pred'], batch['loc_true'], k)
            self.intermediate_result['hit@' + str(k)] += hit
            self.intermediate_result['rank@' + str(k)] += rank
            self.intermediate_result['dcg@' + str(k)] += dcg

    def evaluate(self):
        '''
            通过遍历`self.topk`中的每个 k 值 与 `self.intermediate_result['{hit/rank/dcg}@{k}']`计算以下指标：
            1. **Precision@k**：前 k 个预测中正确预测的比例。
            2. **Recall@k**：前 k 个预测中成功预测真实标签的比例。
            3. **F1@k**：Precision@k 和 Recall@k 的调和平均数。
            4. **MRR@k**：真实标签在前 k 个预测中的平均排名的倒数。
            5. **MAP@k**：真实标签在前 k 个预测中的平均精度。
            6. **NDCG@k**：前 k 个预测的排名质量。
            这些指标共同提供了模型在 top-k 预测中的综合表现评估。

        Returns:
            self.result : Dict[str,float]

            keys:
                Precision@topk
                Recall@topk
                F1@topk
                MRR@topk
                MAP@topk
                NDCG@topk
        '''
        for k in self.topk:
            precision = self.intermediate_result['hit@{}'.format(k)] / (self.intermediate_result['total'] * k)
            self.result['Precision@{}'.format(k)] = precision

            recall = self.intermediate_result['hit@{}'.format(k)] / self.intermediate_result['total']
            self.result['Recall@{}'.format(k)] = recall

            if precision + recall == 0:
                self.result['F1@{}'.format(k)] = 0.0
            else:
                self.result['F1@{}'.format(k)] = (2 * precision * recall) / (precision + recall)

            self.result['MRR@{}'.format(k)] = \
                self.intermediate_result['rank@{}'.format(k)] / self.intermediate_result['total']

            self.result['MAP@{}'.format(k)] = \
                self.intermediate_result['rank@{}'.format(k)] / self.intermediate_result['total']

            self.result['NDCG@{}'.format(k)] = \
                self.intermediate_result['dcg@{}'.format(k)] / self.intermediate_result['total']

            # ---- Acc@k (top-n accuracy / hit rate) ----
            self.result['Acc@{}'.format(k)] = recall  # single-label: top-k acc == hit@k/total

        # ---- Macro metrics & Acc@1 from confusion matrix ----
        if self.cm is None or self.cm.numel() == 0:
            self.result['Acc'] = 0.0
            self.result['MacroRecall'] = 0.0
            self.result['MacroF1'] = 0.0
            return self.result

        cm = self.cm.double()
        total = cm.sum().item()
        if total == 0:
            self.result['Acc'] = 0.0
            self.result['MacroRecall'] = 0.0
            self.result['MacroF1'] = 0.0
            return self.result

        tp = torch.diag(cm)
        support = cm.sum(dim=1)  # true count per class (row sum)
        pred_sum = cm.sum(dim=0)  # predicted count per class (col sum)

        # sklearn default labels = unique_labels(y_true, y_pred)
        mask = (support > 0) | (pred_sum > 0)
        if mask.sum().item() == 0:
            self.result['Acc'] = 0.0
            self.result['MacroRecall'] = 0.0
            self.result['MacroF1'] = 0.0
            return self.result

        # per-class recall / precision with zero_division=0
        recall_c = torch.where(support > 0, tp / support, torch.zeros_like(tp))
        prec_c = torch.where(pred_sum > 0, tp / pred_sum, torch.zeros_like(tp))

        f1_c = torch.where(
            (prec_c + recall_c) > 0,
            2 * prec_c * recall_c / (prec_c + recall_c),
            torch.zeros_like(tp)
        )

        self.result['Acc'] = (tp.sum().item() / total)
        self.result['MacroRecall'] = recall_c[mask].mean().item()
        self.result['MacroF1'] = f1_c[mask].mean().item()

        # 也给 Acc@1 一个别名（可选）
        self.result['Acc@1'] = self.result['Acc']

        return self.result

    def save_result(self, save_path, filename=None):
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:
            filename = str(self.config['exp_id']) + '_' + \
                       datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is {}'.format(json.dumps(self.result, indent=1)))
            path = os.path.join(save_path, '{}.json'.format(filename))
            with open(path, 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at {}'.format(path))

        dataframe = {}
        if 'csv' in self.save_modes:
            for metric in self.metrics:
                dataframe[metric] = []
            for metric in self.metrics:
                for k in self.topk:
                    key_k = metric + '@' + str(k)
                    if key_k in self.result:
                        dataframe[metric].append(self.result[key_k])
                    elif metric in self.result:
                        dataframe[metric].append(self.result[metric])  # repeat
                    else:
                        dataframe[metric].append(np.nan)

            dataframe = pd.DataFrame(dataframe, index=self.topk)
            path = os.path.join(save_path, '{}.csv'.format(filename))
            dataframe.to_csv(path, index=False)
            self._logger.info('Evaluate result is saved at ' + path)
            # 打印完整（避免 pandas 自动省略）
            with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", 200,
                    "display.max_colwidth", None,
            ):
                self._logger.info("\n" + dataframe.to_string())
        return dataframe

    def clear(self):
        '''
        - 清空所有evaluate结果
        - 初始化所有evaluate结果

        Returns:

        '''
        self.result = {}
        # 中间结果
        self.intermediate_result = dict()
        self.intermediate_result['total'] = 0
        # {hit,rank,dcg}@{1,5,10}
        for inter in ['hit']:
            for k in self.topk:
                self.intermediate_result[inter + '@' + str(k)] = 0
        for inter in ['rank', 'dcg']:
            for k in self.topk:
                self.intermediate_result[inter + '@' + str(k)] = 0.0
        # --- for macro metrics / acc ---
        self.num_class = None
        self.cm = None   # confusion matrix: (C, C) long, on CPU

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, accuracy_score, roc_auc_score
def top_n_accuracy(truths, preds, n):
    """ Calculcate Acc@N metric. """
    best_n = np.argsort(-preds, axis=1)[:, :n]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]
def cal_classification_metric(labels, pres):
    """
    Calculates all common classification metrics.

    :param labels: classification label, with shape (N).
    :param pres: predicted classification distribution, with shape (N, num_class).
    """
    pres_index = pres.argmax(-1)  # (N)
    macro_f1 = f1_score(labels, pres_index, average='macro', zero_division=0)
    macro_recall = recall_score(labels, pres_index, average='macro', zero_division=0)
    acc = accuracy_score(labels, pres_index)
    n_list = [5, 10]
    top_n_acc = [top_n_accuracy(labels, pres, n) for n in n_list]

    s = pd.Series([macro_f1, macro_recall, acc] + top_n_acc,
                  index=['macro_f1', 'macro_rec'] +
                  [f'acc@{n}' for n in [1] + n_list])
    return s