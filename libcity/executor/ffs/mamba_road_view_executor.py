import os
import re
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger
from libcity.utils import ensure_dir, get_evaluator
from libcity.executor.bertlm.scheduler import CosineLRScheduler
from libcity.model import loss
from tqdm import tqdm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class MambaRoadViewExecutor(object):
    def __init__(self, config, model, roadgat_data):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.roadgat_data = roadgat_data

        self._logger = getLogger()

        self.vocab_size = self.roadgat_data.get('vocab_size')
        self.driver_num = self.roadgat_data.get('driver_num')

        self.exp_id = self.config.get('exp_id', None)
        self.max_epochs = self.config.get('max_epoch', 100)
        self.model_name = self.config.get('model', '')
        self.line = self.config.get('line')

        self.learner = self.config.get('learner', 'adamw')
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.grad_accmu_steps = self.config.get('grad_accmu_steps', 1)
        self.test_every = self.config.get('test_every', 10)

        self.lr_decay = self.config.get('lr_decay', True)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'cosinelr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.lr_warmup_epoch = self.config.get("lr_warmup_epoch", 5)
        self.lr_warmup_init = self.config.get("lr_warmup_init", 1e-6)
        self.t_in_epochs = self.config.get("t_in_epochs", True)

        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.log_batch = self.config.get('log_batch', 10)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        self.l2_reg = self.config.get('l2_reg', None)

        self.node_features = self.roadgat_data.get('node_features')
        self.edge_index = self.roadgat_data.get('edge_index')
        self.edge_index_trans_prob = self.roadgat_data.get('edge_index_trans_prob')
        self.graph_dict = {
            'node_features': self.node_features,
            'edge_index': self.edge_index,
            'edge_index_trans_prob': self.edge_index_trans_prob,
        }

        # cache ensure
        self.cache_dir = './libcity/cache/{}/{}/model_cache'.format(self.line ,self.exp_id)
        self.png_dir = './libcity/cache/{}/{}'.format(self.line, self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/{}/evaluate_cache'.format(self.line, self.exp_id)
        self.summary_writer_dir = './libcity/cache/{}/{}'.format(self.line, self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.png_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)
        self._writer = SummaryWriter(self.summary_writer_dir)

        self.model = model.to(self.device)
        # 打印基本信息
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(
                param.requires_grad))

        # -- optimizer , lr_scheduler
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self.optimizer.zero_grad()

        #
        self.evaluator = get_evaluator(self.config, self.roadgat_data)
        self.criterion_mask = torch.nn.NLLLoss(ignore_index=0, reduction='none')
        self.initial_ckpt = self.config.get("initial_ckpt", None)
        self.unload_param = self.config.get("unload_param", [])  # 默认空
        if self.initial_ckpt:
            self.load_model_with_initial_ckpt(self.initial_ckpt)


    # --------------------- train, eval(eval on eval_dataloader) , test(eval on testLoader)
    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = -1
        train_time = []
        eval_time = []
        train_loss = []
        train_acc = []
        eval_loss = []
        eval_acc = []
        lr_list = []
        start_epoch = 0


        exp_id = self.config["exp_id"]

        num_batches = len(train_dataloader)
        self._logger.info("Num_batches: train={}, eval={}".format(num_batches, len(eval_dataloader)))

        for epoch_idx in range(start_epoch, self.max_epochs):
            start_time = time.time()
            train_avg_loss, train_avg_acc = self._train_epoch(train_dataloader, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss.append(train_avg_loss)
            train_acc.append(train_avg_acc)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            eval_avg_loss, eval_avg_acc = self._valid_epoch(eval_dataloader, epoch_idx, mode='Eval')
            end_time = time.time()
            eval_time.append(end_time - t2)
            eval_loss.append(eval_avg_loss)
            eval_acc.append(eval_avg_acc)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(eval_avg_loss)
                elif self.lr_scheduler_type.lower() == 'cosinelr':
                    self.lr_scheduler.step(epoch_idx + 1)
                else:
                    self.lr_scheduler.step()

            log_lr = self.optimizer.param_groups[0]['lr']
            lr_list.append(log_lr)
            if (epoch_idx % self.log_every) == 0:
                message = 'Epoch [{}/{}] ({})  train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.max_epochs, (epoch_idx + 1) * num_batches, train_avg_loss,
                           eval_avg_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if eval_avg_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, eval_avg_loss, model_file_name))
                min_val_loss = eval_avg_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

            if (epoch_idx + 1) % self.test_every == 0:
                self.evaluate(test_dataloader)

        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)

        self._draw_png([(train_loss, eval_loss, 'loss'), (train_acc, eval_acc, 'acc'), (lr_list, 'lr')])
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx):
        batches_seen = epoch_idx * len(train_dataloader)

        self.model = self.model.train()

        epoch_loss = []  # total loss of epoch
        total_correct_l = 0  # total top@1 acc for masked elements in epoch
        total_active_elements_l = 0  # total masked elements in epoch

        for i, batch in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            gps_X, gps_padding_mask, \
                road_X, road_Target, road_padding_mask, road_target_masks, road_traj_mat = batch
            _,_, predictions_l = self.model(
                x = road_X,
                padding_masks = road_padding_mask,
                batch_temporal_mat= road_traj_mat,
                graph_dict= self.graph_dict,
                output_hidden_states=False, output_attentions=False
            ) # (B, T, vocab_size)
            targets_l, target_masks_l = road_Target[..., 0], road_target_masks[..., 0] # (B, T), (B, T)
            mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
            # mean_loss_con = self._contrastive_loss(z1, z2, self.contra_loss_type)

            mean_loss = mean_loss_l
            # mean_loss = self.mlm_ratio * mean_loss_l + self.contra_ratio * mean_loss_con
            #
            # if self.test_align_uniform or self.train_align_uniform:
            #     align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
            #     if self.train_align_uniform:
            #         mean_loss += align_uniform_loss

            if self.l2_reg is not None:
                total_loss = mean_loss + self.l2_reg * loss.l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            total_loss = total_loss / self.grad_accmu_steps
            batches_seen += 1

            # with torch.autograd.detect_anomaly():
            total_loss.backward()

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if batches_seen % self.grad_accmu_steps == 0:
                # 调整模型学习参数
                self.optimizer.step()
                if self.lr_scheduler_type == 'cosinelr' and self.lr_scheduler is not None:
                    self.lr_scheduler.step_update(num_updates=batches_seen // self.grad_accmu_steps)
                self.optimizer.zero_grad()

            with torch.no_grad():
                total_correct_l += self._cal_acc(predictions_l, targets_l, target_masks_l)
                total_active_elements_l += num_active_l.item()
                epoch_loss.append(mean_loss.item())  # add total loss of batch

            post_fix = {
                "mode": "Train",
                "epoch": epoch_idx,
                "iter": i,
                "lr": self.optimizer.param_groups[0]['lr'],
                "Loc acc(%)": total_correct_l / total_active_elements_l * 100,
                "MLM loss": mean_loss_l.item(),
                # "Contrastive loss": mean_loss_con.item(),
            }
            # if self.test_align_uniform or self.train_align_uniform:
            #     post_fix['align_loss'] = align_loss
            #     post_fix['uniform_loss'] = uniform_loss
            if i!=0 and i % self.log_batch == 0:
                self._logger.info(str(post_fix))

        epoch_loss = np.mean(epoch_loss)  # average loss per element for whole epoch
        total_correct_l = total_correct_l / total_active_elements_l * 100.0
        self._logger.info("Train: expid = {}, Epoch = {}, avg_loss = {}, total_loc_acc = {}%."
                          .format(self.exp_id, epoch_idx, epoch_loss, total_correct_l))
        self._writer.add_scalar('Train loss', epoch_loss, epoch_idx)
        self._writer.add_scalar('Train loc acc', total_correct_l, epoch_idx)
        return epoch_loss, total_correct_l

    def evaluate(self, test_dataloader):
        """
        use model to test dataset

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        start_time = time.time()
        self. _valid_epoch(test_dataloader, 0, mode='Test')
        t1 = time.time()
        self._logger.info('Test time {}s.'.format(t1 - start_time))

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        self.model = self.model.eval()
        if mode == 'Test':
            self.evaluator.clear()

        epoch_loss = []  # total loss of epoch
        total_correct_l = 0  # total top@1 acc for masked elements in epoch
        total_active_elements_l = 0  # total masked elements in epoch

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), desc="{} epoch={}".format(
                    mode, epoch_idx), total=len(eval_dataloader)):
                gps_X, gps_padding_mask, \
                    road_X, road_Target, road_padding_mask, road_target_masks, road_traj_mat = batch

                _,_, predictions_l = self.model(
                    x = road_X,
                    padding_masks = road_padding_mask,
                    batch_temporal_mat= road_traj_mat,
                    graph_dict= self.graph_dict,
                    output_hidden_states=False, output_attentions=False
                ) # (B, T, vocab_size)
                targets_l, target_masks_l = road_Target[..., 0], road_target_masks[..., 0]
                mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
                # mean_loss_con = self._contrastive_loss(z1, z2, self.contra_loss_type)

                # mean_loss = self.mlm_ratio * mean_loss_l + self.contra_ratio * mean_loss_con
                mean_loss = mean_loss_l

                # if self.test_align_uniform or self.train_align_uniform:
                #     align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
                #     if self.train_align_uniform:
                #         mean_loss += align_uniform_loss

                if mode == 'Test':
                    evaluate_input = {
                        'loc_true': targets_l[target_masks_l].reshape(-1, 1).squeeze(-1).cpu().numpy(),  # (num_active, )
                        'loc_pred': predictions_l[target_masks_l].reshape(-1, predictions_l.shape[-1]).cpu().numpy() # (num_active, n_class)
                    }
                    self.evaluator.collect(evaluate_input)

                total_correct_l += self._cal_acc(predictions_l, targets_l, target_masks_l)
                total_active_elements_l += num_active_l.item()
                epoch_loss.append(mean_loss.item())  # add total loss of batch

                post_fix = {
                    "mode": "Train",
                    "epoch": epoch_idx,
                    "iter": i,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "Loc acc(%)": total_correct_l / total_active_elements_l * 100,
                    "MLM loss": mean_loss_l.item(),
                    # "Contrastive loss": mean_loss_con.item(),
                }
                # if self.test_align_uniform or self.train_align_uniform:
                #     post_fix['align_loss'] = align_loss
                #     post_fix['uniform_loss'] = uniform_loss
                if i!=0 and i % self.log_batch == 0:
                    self._logger.info(str(post_fix))

            epoch_loss = np.mean(epoch_loss)  # average loss per element for whole epoch
            total_correct_l = total_correct_l / total_active_elements_l * 100.0
            self._logger.info("{}: expid = {}, Epoch = {}, avg_loss = {}, total_loc_acc = {}%."
                              .format(mode, self.exp_id, epoch_idx, epoch_loss, total_correct_l))
            self._writer.add_scalar('{} loss'.format(mode), epoch_loss, epoch_idx)
            self._writer.add_scalar('{} loc acc'.format(mode), total_correct_l, epoch_idx)

            if mode == 'Test':
                self.evaluator.save_result(self.evaluate_res_dir)
            return epoch_loss, total_correct_l

    # --------------------- loss
    def _cal_loss(self, pred, targets, targets_mask):
        '''

        Parameters
        ----------
        pred (B, T, V)
        targets (B, T)
        targets_mask(B, T)

        Returns
        -------

        '''
        batch_loss_list = self.criterion_mask(pred.transpose(1, 2), targets.long())
        batch_loss = torch.sum(batch_loss_list)
        num_active = targets_mask.sum()
        mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization
        return mean_loss, batch_loss, num_active

    def _cal_acc(self, pred, targets, targets_mask):
        '''

        Parameters
        ----------
        pred (B, T, V)
        targets (B, T)
        targets_mask(B, T)

        Returns
        -------

        '''
        mask_label = targets[targets_mask].long()  # (num_active, )
        lm_output = pred[targets_mask].argmax(dim=-1)  # (num_active, )
        correct_l = mask_label.eq(lm_output).sum().item()
        return correct_l

    def align_loss(self, x, y, alpha=2):
        if self.norm_align_uniform:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        if self.norm_align_uniform:
            x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def align_uniform(self, x, y):
        align_loss_val = self.align_loss(x, y, alpha=self.align_alpha)
        unif_loss_val = (self.uniform_loss(x, t=self.unif_t) + self.uniform_loss(y, t=self.unif_t)) / 2
        sum_loss = align_loss_val * self.align_w + unif_loss_val * self.unif_w
        return sum_loss, align_loss_val.item(), unif_loss_val.item()

    def _contrastive_loss(self, z1, z2, loss_type):
        if loss_type == 'simsce':
            return self._contrastive_loss_simsce(z1, z2)
        elif loss_type == 'simclr':
            return self._contrastive_loss_simclr(z1, z2)
        elif loss_type == 'consert':
            return self._contrastive_loss_consert(z1, z2)
        else:
            raise ValueError('Error contrastive loss type {}!'.format(loss_type))

    def _contrastive_loss_simclr(self, z1, z2):
        """

        Args:
            z1(torch.tensor): (batch_size, d_model)
            z2(torch.tensor): (batch_size, d_model)

        Returns:

        """
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        if self.similarity == 'inner':
            similarity_matrix = torch.matmul(features, features.T)
        elif self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)  # (batch_size * 2, 1)
        logits = logits / self.temperature

        loss_res = self.criterion(logits, labels)
        return loss_res

    def _contrastive_loss_simsce(self, z1, z2):
        assert z1.shape == z2.shape
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        if self.similarity == 'inner':
            similarity_matrix = torch.matmul(z1, z2.T)
        elif self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(z1, z2.T)
        similarity_matrix /= self.temperature

        labels = torch.arange(similarity_matrix.shape[0]).long().to(self.device)
        loss_res = self.criterion(similarity_matrix, labels)
        return loss_res

    # --------------------- Infrastructure
    def _draw_png(self, data):
        '''
        [(train_loss, eval_loss, 'loss'), (train_acc, eval_acc, 'acc'), (lr_list, 'lr')]
        train_loss : list len=$x$,
        eval_loss : list len=$x$,
        lr : titlename
        '''
        for data_iter in data:
            plt.figure()
            if len(data_iter) == 3:
                # 两条曲线，一个title name
                train_list, eval_list, name = data_iter
                x_index = np.arange((len(train_list)))
                plt.plot(x_index, train_list, 'r', label='train_{}'.format(name))
                plt.plot(x_index, eval_list, 'b', label='eval_{}'.format(name))
            else:
                # 一条曲线，一个title name
                data_list, name = data_iter
                x_index = np.arange((len(data_list)))
                plt.plot(x_index, data_list, 'r', label='{}'.format(name))
            plt.ylabel(name)
            plt.xlabel('epoch')
            plt.title(str(self.exp_id) + ': ' + str(self.model_name))
            plt.legend()
            path = self.png_dir + '/{}_{}.png'.format(self.exp_id, name)
            plt.savefig(path)
            self._logger.info('Save png at {}'.format(path))

    def _valid_parameter(self, k):
        '''

        Args:
            k: List

        Returns:
            unload_param中的参数在k中存在， 返回True
            unload_param中的所有参数在k中均不存在，返回False

            如果k为str
            - k为unload_param中的参数， 返回True
            - 否则，False
        '''
        for para in self.unload_param:
            if para in k:
                return True
        return False

    def load_model_with_initial_ckpt(self, initial_ckpt):
        '''
        加载预训练模型
        - 从预训练模型权重文件initial_ckpt中加载参数到当前模型
        - 处理 model和 initial_ckpt获得的pretrained_model 相交的参数，加载到model中

        Args:
            initial_ckpt: path, 初始模型位置，继续训练

        Returns:

        '''
        assert os.path.exists(initial_ckpt), 'Weights at %s not found' % initial_ckpt
        checkpoint = torch.load(initial_ckpt, map_location='cpu')
        pretrained_model = checkpoint['model'].state_dict()
        # 当前模型的state_dict
        model_keys = self.model.state_dict()
        state_dict_load = {}
        unexpect_keys = []
        # 将pretrain_model中和model匹配的kv加入state_dict_load
        # pretrain ∩ model
        for k, v in pretrained_model.items():
            # 不在当前模型的keys中，或者维度不匹配，或者需要unload的参数
            if k not in model_keys.keys() or v.shape != model_keys[k].shape\
                    or self._valid_parameter(k):
                unexpect_keys.append(k)
            else:
                state_dict_load[k] = v
        # 差集
        # unexcept_keys = (pretrain_model - model) ∪ (model-pretrain_model)
        for k, v in model_keys.items():
            if k not in pretrained_model.keys():
                unexpect_keys.append(k)
        self._logger.info("Unexpected keys: {}".format(unexpect_keys))
        self.model.load_state_dict(state_dict_load, strict=False)
        self._logger.info("Initialize model from {}".format(initial_ckpt))

    def save_model(self, cache_name):
        """
        保存模型

        Args:
            cache_name(str): 保存的文件名
                # pipeline # ./libcity/cache/{}/model_cache/{}_{}_{}.pt
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model'] = self.model.cpu()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(config, cache_name)
        self.model.to(self.device)
        self._logger.info("Saved model at " + cache_name)

    def load_model_state(self, cache_name):
        """
        加载对应模型的 cache （用于加载参数直接进行测试的场景）

        Args:
            cache_name(str): 保存的文件名
                # pipeline # ./libcity/cache/{}/model_cache/{}_{}_{}.pt
        """
        assert os.path.exists(cache_name), 'Weights at {} not found' % cache_name
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'].state_dict()) # 加载到当前model
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at " + cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        assert os.path.exists(cache_name), 'Weights at {} not found' % cache_name
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model = checkpoint['model'].to(self.device) # 直接加载预训练模型，覆盖model
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at " + cache_name)

    def save_model_with_epoch(self, epoch):
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        ensure_dir(self.cache_dir) # './libcity/cache/{}/model_cache'.format(self.exp_id)
        config = dict()
        config['model'] = self.model.cpu()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self.model.to(self.device)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw': #
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                          eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            elif self.lr_scheduler_type.lower() == 'cosinelr': #
                lr_scheduler = CosineLRScheduler(
                    self.optimizer, t_initial=self.max_epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                    warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init, t_in_epochs=self.t_in_epochs)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

