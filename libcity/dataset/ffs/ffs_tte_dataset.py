import json
import math
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from libcity.dataset.pm.road_vocab import RoadVocab
from libcity.utils.utils import ensure_dir
import pandas as pd
from tqdm import tqdm
from logging import getLogger
from einops import repeat, rearrange
import pickle
# from concurrent.futures import ProcessPoolExecutor
# import datetime


class FFSTTE_Dataset:
    def __init__(self, config):
        '''FFSTTEDataset 只组织gps的dataset
        '''
        self._logger = getLogger()
        self.config = config

        # -- basic
        self.line = self.config.get("line", 'ffsmamba')
        self.dataset = self.config.get('dataset', 'chengdu')
        self.batch_size = self.config['batch_size']  # 同步GPS和Road的batch_size
        self.device = self.config['device']
        self.seq_len = self.config['seq_len']
        self.num_workers = self.config.get('num_workers', 0)
        self.add_cls_for_road = config.get('add_cls_for_road', False)
        self.add_cls_for_poi = config.get('add_cls_for_poi', False)

        self.traj_path = self.config.get("traj_path", None)
        self.traj_train_path = self.traj_path[:-8] + "_train.parquet"
        self.traj_eval_path = self.traj_path[:-8] + "_eval.parquet"
        self.traj_test_path = self.traj_path[:-8] + "_test.parquet"
        self.cache_path = f"./raw_data/{self.line}/{self.dataset}/cache/"
        ensure_dir(self.cache_path)

        # -- road view
        # self.vocab_path = self.config.get('vocab_path', None)
        # self.road_path = self.config.get('road_path', None)
        # self.road_meta_path = self.config.get('road_meta_path', None)
        # self.rel_path = self.config.get('rel_path', None)

        # -- -- 加载road vocab
        # self.driver_num = 0
        # self.vocab_size = 0
        # self.vocab = None
        # self.__load_vocab()

        # -- --  准备和roadGat相关的内容
        # self.road_df = None
        # self.rel_df = None
        # self.road_size = None
        # self.node_features = None  # (vocab_size, node_feature_dim)
        # self.node_fea_dim = 0  # 路段维度
        # self.edge_index = None  # 边索引 (2, E), rel_df的转置
        # self.edge_index_trans_prob = None  # 路段转移索引
        # self.roadgat_neighbor_path = self.config.get('roadgat_neighbor_path', None)
        # self.roadgat_transprob_path = self.config.get('roadgat_transprob_path', None)
        # self.__prepare_roadgat()

        # -- -- span-mlm
        # self.masking_ratio = self.config.get('masking_ratio', 0.15)
        # self.avg_mask_len = self.config.get('avg_mask_len', 2)
        # self.masking_mode = self.config.get('masking_mode', 'together')
        # self.distribution = self.config.get('distribution', None)

        # -- dataset 2 dataloader
        # self.collate_fn = PretrainCollateFn(max_len=self.seq_len, vocab=self.vocab,
        #                                          add_cls_for_road=self.add_cls_for_road,
        #                                          add_cls_for_poi=self.add_cls_for_poi)
        self.collate_fn = MambaFuseViewCollateFn(config = self.config)
    
    '''
    Remove functions .. 
        __load_vocab
        __prepare_roadgat
        __get_roadgat_data
    '''

    def get_data(self):
        # Usage: 供外部调用，直接获取dataloader
        self._logger.info("🔄 生成 Dataset!")
        train_dataset, eval_dataset, test_dataset = self.__gen_dataset()
        self._logger.info('📈 Size of dataset[Train Eval Test]: ' +
                          str(len(train_dataset)) + '/' + str(len(eval_dataset)) + '/' + str(len(test_dataset)))
        self._logger.info("🔄 生成 Dataloader!")
        return self.__gen_dataloader(train_dataset, eval_dataset, test_dataset)

    def __gen_dataset(self):
        train_dataset = FFSTTEInnerDataset(
            config=self.config,
            type='train',
        )
        eval_dataset = FFSTTEInnerDataset(
            config=self.config,
            type='eval',
        )
        test_dataset = FFSTTEInnerDataset(
            config=self.config,
            type='test',
        )
        return train_dataset, eval_dataset, test_dataset

    def __gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        '''
        ->  collate_fn
        Args:
            train_dataset:
            eval_dataset:
            test_dataset:

        Returns:

        '''
        assert self.collate_fn is not None
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=True,
                                      collate_fn=lambda raw_batch: self.collate_fn(raw_batch))
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=True,
                                     collate_fn=lambda raw_batch: self.collate_fn(raw_batch))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     shuffle=False,
                                     collate_fn=lambda raw_batch: self.collate_fn(raw_batch))
        return train_dataloader, eval_dataloader, test_dataloader

class FFSTTEInnerDataset(Dataset):
    def __init__(self, config, type):
        self._logger = getLogger()
        self.config = config
        self.type = type

        self.traj_data_path = f"{self.config['traj_path'][:-8]}_{self.type}.parquet"
        self.cache_path = f"./raw_data/{self.config['line']}/{self.config['dataset']}/cache/"
        ensure_dir(self.cache_path)

        # -- 加载/保存gpsview，roadview，road_mat 的cache path
        self.gps_traj_list_path = self.cache_path + f"MambaFuseViewInnerDataset_GpsTrajList_{self.type}.pkl"
        # self.road_traj_list_path = self.cache_path + f"MambaFuseViewInnerDataset_RoadTrajList_{self.type}.pkl"
        # self.road_traj_mat_list_path = self.cache_path + f"MambaFuseViewInnerDataset_RoadTrajMatList_{self.type}.pkl"
        self.use_cache = config.get("use_cache", False)
        self.cache_gps_traj_list_path = config.get("cache_gps_traj_list_path")
        self.cache_gps_traj_list_path = self.cache_gps_traj_list_path[:-4] + f"_{type}.pkl"

        # -- road view span-mlm
        # self.vocab = vocab
        # self.masking_ratio = self.config.get('masking_ratio', 0.15)
        # self.avg_mask_len = self.config.get('avg_mask_len', 2)
        # self.masking_mode = self.config.get('masking_mode', 'together')
        # self.distribution = self.config.get('distribution', None)

        # -- 加载
        self.gps_traj_list = None
        # self.road_traj_list = None
        # self.road_traj_mat_list = None
        self._load_data()

    def _load_data(self):
        '''
        加载 GpsView & RoadView

        - gps_traj_list  List, item: ndarr(len_max_gps, f_gps)
        # - road_traj_list List, item: ndarr(len_max_road, f_road)
        # - road_traj_mat_list List, item: ndarr(len_max_road, len_max_road)

        '''
        if self.use_cache:
            self._logger.info("🤗 Use Cache 20w here")
            self.gps_traj_list = pickle.load(open(self.cache_gps_traj_list_path, 'rb'))
            self._logger.info(f"🤗 Use Cache 20w here: gps_traj_list loaded from {self.cache_gps_traj_list_path}")
        else:
            if os.path.exists(self.gps_traj_list_path):
                self.gps_traj_list = pickle.load(open(self.gps_traj_list_path, 'rb'))
                # self.road_traj_list = pickle.load(open(self.road_traj_list_path, 'rb'))
                # self.road_traj_mat_list = pickle.load(open(self.road_traj_mat_list_path, 'rb'))
            else:
                self.gps_traj_list= self.data_processing()
                pickle.dump(self.gps_traj_list, open(self.gps_traj_list_path, 'wb'))

    def __getitem__(self, index):
        gps_traj = self.gps_traj_list[index] # nd (len_gps_traj, f_traj)
        # road_traj = self.road_traj_list[index] # (len_road_traj, f_road)
        # road_traj_mat = self.road_traj_mat_list[index] # (len_road_traj, len_road_traj)
        # (len_road_traj, f-road)
        # road_span_mlm_mask = noise_mask(road_traj, self.masking_ratio, self.avg_mask_len, self.masking_mode, exclude_feats=None, add_cls=False)
        return gps_traj # , road_traj, road_traj_mat, road_span_mlm_mask

    def __len__(self):
        return len(self.gps_traj_list)

    def data_processing(self):
        self._logger.info(f"🔄 MambaFuseViewInnerDataset#data_processing: {self.traj_data_path} {self.type}: ")
        # (len_gps_traj, ...)
        origin_df = pd.read_parquet(self.traj_data_path, engine='fastparquet')
        gps_traj_list = self.data_processing_for_gps(origin_df)
        # road_traj_list, road_traj_mat_list = self.data_processing_for_road(origin_df)
        return gps_traj_list # , road_traj_list, road_traj_mat_list


    def data_processing_for_gps(self, origin_df):
        """
        Returns
        ----------
        gps_traj_list: list(ndarr)
            ndarr: (len_gps_traj, F), F = (tm, delta_tm, lng, lat, speed, acc, angle_delta)
                    其中，除了tm和delta_tm之外，其余全部特征都被normalized
        """
        self._logger.info(f"🔄 MambaFuseViewInnerDataset # data_processing_for_gps")
        #
        # # (len_gps_traj, ...)
        # origin_df = pd.read_parquet(self.traj_data_path, engine='fastparquet')
        gps_traj_list = []
        """ 关于gps序列的特征
        # basic
        'gps_tm_list',
        'gps_lat_list', 'gps_lng_list', 
        'gps_speed_list', 'gps_acceleration_list', 'gps_angle_delta_list', 

        # other ... 
        'gps_road_list', 
        'gps_interval_list', 'gps_dist_list', 
        """

        lng_list = [lng for lng_list in origin_df['gps_lng_list'] for lng in lng_list]
        lat_list = [lat for lat_list in origin_df['gps_lat_list'] for lat in lat_list]

        max_lng, min_lng = max(lng_list), min(lng_list)
        max_lat, min_lat = max(lat_list), min(lat_list)

        gps_traj_list = []

        for i in tqdm(range(math.floor(origin_df.shape[0])), desc='MambaFuseViewInnerDataset: 处理GPSView ...'):
            one_traj = origin_df.iloc[i]

            # -- timestamp, delta_time
            one_gps_tm_list = np.array(one_traj['gps_tm_list'])
            new_gps_tm_list = [pd.to_datetime(tm, unit='s') for tm in one_gps_tm_list]
            start_time = pd.to_datetime(one_traj['start_time'], unit='s')
            gps_delta_time_list = [(tm - start_time).total_seconds() for tm in new_gps_tm_list]  #
            one_gps_minute_list = one_gps_tm_list % (60 * 60)
            one_gps_hour_list = one_gps_tm_list % (24 * 60 * 60) / (60 * 60)
            one_gps_week_list = one_gps_tm_list % (7 * 24 * 60 * 60) / (24 * 60 * 60)

            # -- lng, lat
            one_gps_lng_list = one_traj['gps_lng_list']
            one_gps_lat_list = one_traj['gps_lat_list']
            # norm lng, lat
            one_gps_lng_list = (np.array(one_gps_lng_list, dtype='float') - min_lng) / (max_lng - min_lng)
            one_gps_lat_list = (np.array(one_gps_lat_list, dtype='float') - min_lat) / (max_lat - min_lat)

            # -- speed, acc, course_angle,
            one_gps_speed_list = one_traj['gps_speed_list']
            one_gps_acceleration_list = one_traj['gps_acceleration_list']
            one_gps_angle_delta_list = one_traj['gps_angle_delta_list']
            # fill_none
            one_gps_speed_list[0] = one_gps_speed_list[1]
            one_gps_acceleration_list[0] = one_gps_acceleration_list[2]
            one_gps_acceleration_list[1] = one_gps_acceleration_list[2]
            one_gps_angle_delta_list[0] = one_gps_angle_delta_list[2]
            one_gps_angle_delta_list[1] = one_gps_angle_delta_list[2]
            # norm
            speed_max, speed_min = max(one_gps_speed_list), min(one_gps_speed_list)
            acc_max, acc_min = max(one_gps_acceleration_list), min(one_gps_acceleration_list)
            delta_max, delta_min = max(one_gps_angle_delta_list), min(one_gps_angle_delta_list)
            one_gps_speed_list = (np.array(one_gps_speed_list, dtype='float') - speed_min) / (speed_max - speed_min)
            one_gps_acceleration_list = (np.array(one_gps_acceleration_list, dtype='float') - acc_min) / (
                        acc_max - acc_min)
            one_gps_angle_delta_list = (np.array(one_gps_angle_delta_list, dtype='float') - delta_min) / (
                        delta_max - delta_min)

            traj_fea = np.array(
                [
                    one_gps_tm_list, gps_delta_time_list, one_gps_minute_list, one_gps_hour_list, one_gps_week_list,
                    one_gps_lng_list, one_gps_lat_list,
                    one_gps_speed_list, one_gps_acceleration_list, one_gps_angle_delta_list,
                ]
            ).transpose((1, 0))  # (7, len_max_gps) -> (len_max_gps, 7)
            gps_traj_list.append(traj_fea)
        return gps_traj_list



class MambaFuseViewCollateFn:
    def __init__(self, config):
        self.config = config
        # self.vocab = vocab

        self.device = self.config['device']
        # self.add_cls_for_road = self.config['add_cls_for_road']
        self.add_cls_for_gps = self.config['add_cls_for_gps']
        self.seq_len = self.config['seq_len']
        self.max_len = self.seq_len


    def __call__(self, raw_batch):
        '''处理raw_batch 为批量

        Args:
            raw_batch: List[ (gps_traj, road_traj), ... ]
                gps_traj: ndarray(len_gps_traj, f_gps)
                road_traj: ndarray(len_road_traj, f_road)
                road_traj_mat: ndarray(len_road_traj, len_road_traj)
                road_span_mlm_mask: ndarray(len_road_traj, f_road)

        Returns: tensor.device
            gps_X tensor(batch_size, max_gpslen, fgps)  使用vocab.pad_index:0 遮盖padding部分
            road_X tensor(batch_size, max_roadlen, froad) 使用vocab.pad_index:0 遮盖padding部分
            gps_padding_mask tensor(batch_size, max_gpslen) True保留，False遮盖
            road_padding_mask tensor(batch_size, max_roadlen) True保留，False遮盖
        '''
        # gps_traj_batch, road_traj_batch, road_traj_mat_batch, road_span_mlm_mask_batch = [], [], [], []
        # for row in raw_batch:
        #     gps_traj_batch.append(row[0])
        #     road_traj_batch.append(row[1])
        #     road_traj_mat_batch.append(row[2])
        #     road_span_mlm_mask_batch.append(row[3])

        gps_traj_batch = raw_batch


        '''Road View
        road_X.to(self.device), # (batch_size, len_max_road, f_road) maskIndex遮盖vocab_road_id，padIndex遮盖特征
        road_Target.to(self.device), # (batch_size, len_max_road, f_road) 遮盖的vocab_road_id和feature真值, 其余为pad_index
        road_padding_mask.to(self.device), # (batch_size, len_max_road)
        road_target_masks.to(self.device), #  (batch_size, len_max_road, f_road) True遮盖，False保留
        road_traj_mat.to(self.device), # (batch_size, len_max_road, len_max_road)
        '''
        # road_X, road_Target, road_padding_mask, road_target_masks, road_traj_mat = self.__call_for_road__(road_traj_batch, road_traj_mat_batch, road_span_mlm_mask_batch)


        '''
        gps_X.to(self.device), # (batch_size, len_max_gps, f_gps)
        gps_padding_mask.to(self.device) # (batch_size, len_max_gps)
        gps_tte_targets.to(self.device) # (batch_size, 1)
        '''
        gps_X, gps_padding_mask, gps_tte_targets = self.__call_for_gps__(gps_traj_batch)

        return (
            gps_X, # (batch_size, len_max_gps, f_gps)
            gps_padding_mask, # (batch_size, len_max_gps), True遮盖，False保留
            gps_tte_targets, # (batch_size, 1) gps轨迹的tte
            # road_X, # (batch_size, len_max_road, f_road) maskIndex遮盖vocab_road_id，padIndex遮盖特征
            # road_Target, # (batch_size, len_max_road, f_road) 遮盖的vocab_road_id和feature真值, 其余为pad_index
            # road_padding_mask, # (batch_size, len_max_road)
            # road_target_masks, #  (batch_size, len_max_road, f_road) True遮盖，False保留
            # road_traj_mat # (batch_size, len_max_road, len_max_road)
        )


    def __call_for_road__(self, road_traj_batch, road_traj_mat_batch, road_span_mlm_mask_batch):
        ''' 处理raw_batch 为批量

        Args:
            raw_batch: List[ (gps_traj, road_traj), ... ]
                gps_traj: ndarray(len_gps_traj, f_gps)
                road_traj: ndarray(len_road_traj, f_road)
                road_traj_mat: ndarray(len_road_traj, len_road_traj)
                road_span_mlm_mask: ndarray(len_road_traj, f_road)

        Returns: tensor.device
            gps_X tensor(batch_size, max_gpslen, fgps)  使用vocab.pad_index:0 遮盖padding部分
            road_X tensor(batch_size, max_roadlen, froad) 使用vocab.pad_index:0 遮盖padding部分
            gps_padding_mask tensor(batch_size, max_gpslen) True保留，False遮盖
            road_padding_mask tensor(batch_size, max_roadlen)

        '''
        # -- 将data: List[ tuple(gps, road), ...] 重新组织为gps_batch, road_batch
        # -- -- 1、zip(*data)
        # gps_trajs, road_trajs = zip(*raw_batch)
        # gps_trajs tuple( gps_traj )
        # road_trajs tuple( road_traj )

        # -- -- 2、list add
        # gps_traj_batch list[ gps_traj ], gps_traj: (len_gps_traj, f_gps)...
        # road_traj_batch list[ road_traj ], road_traj

        # gps_traj_batch, road_traj_batch, road_traj_mat_batch, road_span_mlm_mask_batch = [], [], [], []
        # for row in raw_batch:
        #     gps_traj_batch.append(row[0])
        #     road_traj_batch.append(row[1])
        #     road_traj_mat_batch.append(row[2])
        #     road_span_mlm_mask_batch.append(row[3])
        batch_size = len(road_traj_batch)
        # gps_valid_lengths = [ x.shape[0] for x in gps_traj_batch ]
        road_valid_lengths = [ x.shape[0] for x in road_traj_batch ]
        # max_gps_len = max(gps_valid_lengths)
        max_road_len = max(road_valid_lengths)

        # gps_X = torch.zeros(batch_size, max_gps_len, gps_traj_batch[0].shape[-1], dtype=torch.float32)
        road_X = torch.zeros(batch_size, max_road_len, road_traj_batch[0].shape[-1], dtype=torch.float32)
        road_X.fill_(self.vocab.pad_index) # 默认用0遮盖
        road_target_masks = torch.zeros_like(road_X, dtype=torch.bool)
        road_traj_mat = torch.zeros(batch_size, max_road_len, max_road_len, dtype=torch.float32)
        for i in range(batch_size):
            # gps_end = min(gps_valid_lengths[i], max_gps_len)
            # (gps_traj_len, f)
            # gps_X[i, :gps_end, :] = torch.tensor(gps_traj_batch[i][:gps_end, :], dtype=torch.float32)

            road_end = min(road_valid_lengths[i], max_road_len)
            road_X[i, :road_end, :] = torch.tensor(road_traj_batch[i][:road_end, :], dtype=torch.float32)
            road_target_masks[i, :road_end, :] = torch.tensor(road_span_mlm_mask_batch[i][:road_end, :], dtype=torch.bool)
            road_traj_mat[i, :road_end, :road_end] = torch.tensor(road_traj_mat_batch[i][:road_end, :road_end], dtype=torch.float32)


        # (batch_size, len_max_gps)
        # gps_padding_mask = padding_mask(torch.tensor(gps_valid_lengths, dtype=torch.int16), max_len=max_gps_len)
        # (batch_size, len_max_road)
        road_padding_mask = padding_mask(torch.tensor(road_valid_lengths, dtype=torch.int16), max_len=max_road_len)

        # -- 对road 作 span-mlm？
        # -- -- road_span_mlm_mask_batch True保留， False span-mlm遮盖
        # -- -- road_padding_mask True保留， False遮盖
        # -- -- 最终：True  未padding的span-mlm遮盖部分
        # -- --      False (1) (True)span-mlm遮盖*(False)无效长度, (2) (False)span-mlm未遮盖*(True/False)有效长度/无效长度
        # -- -- ~road_target_masks * road_padding_mask.unsqueeze(-1),
        # -- -- True部分=span-mlm遮盖部分*有效长度内的padding
        # -- -- False部分= span-mlm未遮盖*[有效长度、无效长度]
        # (batch_size, len_max_road, f_road)
        road_target_masks = ~road_target_masks
        road_target_masks = road_target_masks * road_padding_mask.unsqueeze(-1)

        # 保留span-mlm遮盖的真实部分，作为labels/targets
        # road_target_masks True表示span-mlm部分
        road_Target = road_X.clone()
        road_Target = road_Target.masked_fill_(road_target_masks == 0, self.vocab.pad_index)

        # 构建输入
        # gps_X.masked_fill_(gps_padding_mask.unsqueeze(-1) ==0, self.vocab.pad_index) # 默认 gpsX的所有特征 都用0遮盖
        # road_X[..., 0:1]: (batch_size, len_max_road, 1), f_road的第一个就是路段特征
        # road_X[..., 0]: (batch_size, len_max_road), ❌
        # 可见：test.mytorch.torchapis.testDotDotDot
        road_X[..., 0:1].masked_fill_(road_target_masks[...,0:1] == 1, self.vocab.mask_index) # mask
        road_X[...,1:].masked_fill_(road_target_masks[...,1:] == 1, self.vocab.pad_index)

        # gps_traj_batch = torch.from_numpy(pad_batch(gps_traj_batch)).float().to(self.device)
        # road_traj_batch = torch.from_numpy(pad_batch(road_traj_batch)).float().to(self.device)



        return (
            # -- gps
            # gps_X.to(self.device), # (batch_size, len_max_gps, f_gps)
            # gps_padding_mask.to(self.device),  # (batch_size, len_max_gps, len_max_gps)
            # -- road
            road_X.to(self.device), # (batch_size, len_max_road, f_road) maskIndex遮盖vocab_road_id，padIndex遮盖特征
            road_Target.to(self.device), # (batch_size, len_max_road, f_road) 遮盖的vocab_road_id和feature真值, 其余为pad_index
            road_padding_mask.to(self.device), # (batch_size, len_max_road)
            road_target_masks.to(self.device), #  (batch_size, len_max_road, f_road) True遮盖，False保留
            road_traj_mat.to(self.device), # (batch_size, len_max_road, len_max_road)
        )


    def __call_for_gps__(self, gps_traj_batch):
        '''处理GpsView的raw_batch

        Parameters
        ----------
        raw_batch: List[ gps_traj ]
            gps_traj (len_gps, F)
                ( gps_tm, gps_delta_time, gps_lng, gps_lat, gps_speed, gps_acc, gps_angle_delta )
            ...

        Retuns
        ------
        两个gps数据
        gps_X (batch_size, len_gps, F)
            gps_X[:, :, i] \in (batch_size, len_gps, 1)
            0 gps_tm
            1 gps_delta_time
            2 gps_min
            3 gps_hour
            4 gps_week
            ---
            5 gps_lng
            6 gps_lat
            ---
            7 gps_speed
            8 gps_acc
            9 gps_angle_delta
        gps_padding_mask (batch_size, len_gps) True保留，False遮盖
        '''
        # gps_traj_batch = []
        # for row in raw_batch:
        #     gps_traj_batch.append(row)
        batch_size = len(gps_traj_batch)
        gps_valid_lengths = [ x.shape[0] for x in gps_traj_batch]
        max_gps_len = max(gps_valid_lengths)

        # -- make batch, pad with 0
        # (batch_size, len_max_gps, f)
        gps_X = torch.zeros(batch_size, max_gps_len, gps_traj_batch[0].shape[-1], dtype=torch.float32)
        # -- gps_tte_targets
        gps_tte_targets = []
        gps_v_stalls = torch.zeros(batch_size, max_gps_len) # (B, T)
        for i in range(batch_size):
            gps_end = min(gps_valid_lengths[i], max_gps_len)
            # len_max_gps, f
            gps_X[i, :gps_end, :] = torch.tensor(gps_traj_batch[i][:gps_end, :], dtype=torch.float32)
            gps_v = torch.tensor(gps_traj_batch[i][:gps_end, 7]) # (len_gps,)
            gps_v_mean = gps_v.mean()
            gps_v_stall = torch.relu(gps_v_mean - gps_v) # (len_gps,)
            gps_v_stalls[i, :gps_end] = gps_v_stall
            gps_tte_targets.append(gps_traj_batch[i][gps_end-1,0] - gps_traj_batch[i][0,0])

        # (batch_size, len_max_gps)
        gps_padding_mask = padding_mask(torch.tensor(gps_valid_lengths, dtype=torch.int16), max_len=max_gps_len)
        # 默认 gpsX的所有特征 都用0遮盖
        gps_X.masked_fill_(gps_padding_mask.unsqueeze(-1) == 0, self.config['pad_index'])

        # tte_targets
        gps_tte_targets = torch.tensor(gps_tte_targets, dtype=torch.float32).reshape(-1,1)

        # -- for tte
        # 每条gps轨迹，从第2个gps点，开始遮盖tm
        gps_X[:, 1:, 0:5] = 0
        gps_X[:, 1:, 7:9] = 0 # 9 转角可以保留

        gps_v_stalls = gps_v_stalls.unsqueeze(-1) # (B,L,1)
        gps_X = torch.cat([gps_X, gps_v_stalls], dim=-1)

        return (
            # -- gps
            gps_X.to(self.device), # (B, len_max_gps, f_gps)
            gps_padding_mask.to(self.device), #(B, len_max_gps)
            gps_tte_targets.to(self.device), # (B,1)
        )





'''
Helper Function:
 - padding_mask
 - nosie_mask
    - geom_noise_mask_single
'''

def padding_mask(lengths, max_len=None):
    '''
    假如轨迹长度为2, max_len=3, 轨迹pad至3, 则padding_mask就是
    [1 1 0]
    True保留，False遮盖

    Args:
        lengths: List[ int ], lengths[i]: i-th轨迹的长度
        max_len: padding
    Returns:
        padding_mask: (batch_size, max_len or max(lengths))

    '''

    batch_size = lengths.numel()
    # max_len不为空，使用max_len; 否则使用lengths最大值
    # trick works because of overloading of 'or' operator for non-boolean types
    max_len = max_len or lengths.max_val()
    # TODO note here: or短路判断; padding_mask
    '''
    aranges: torch.arange(0, max_len, device=lengths.device).type_as(lengths) # (max_len)
        repeat(batch_size,1) # (batch_size,max_len)
    lengths: lengths.unsqueeze(1) # (batch_size,1)

    aranges < lengths
        True 保留
        False mask
    '''
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def noise_mask(X, masking_ratio, lm=3, mode='together', distribution='random', exclude_feats=None, add_cls=True):
    '''单条轨迹的特征遮盖，连续遮盖masking_ratio比例的轨迹。False遮盖, True保留

    Args:
        X: traj_cls  (trajPer_cls_len, feat_dim)
        masking_ratio:
        lm:  avg_mask_len
        mode: masking_mode
        distribution:  'geometric'
        exclude_feats:  None
        add_cls: True
    Returns:
        mask: ndarray(traj_len, feat_dim) True不遮盖，False遮盖

    '''
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    # 使用几何分布生成掩码
    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':
            # (trajPer_cls_len,F)
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)
        else: # together
            mask = repeat(
                geom_noise_mask_single(X.shape[0], lm, masking_ratio),
                "traj_cls_len->traj_cls_len feature_dim",
                feature_dim=X.shape[1]
            )
    elif distribution == 'random':  # each position is independent Bernoulli with p = 1 - masking_ratio
        # 伯努利分布，以masking_ratio的概率置做mask
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            # (traj_cls_len,1) -> (traj_cls_len, feature_dim)
            one_col_span_mlm_mask = np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,p=(1 - masking_ratio, masking_ratio))
            mask = repeat( one_col_span_mlm_mask,
                "traj_cls_len 1 -> traj_cls_len feature_dim", feature_dim=X.shape[1]
            )
    else:
        # 不指定distribution，就不遮盖
        mask = np.ones(X.shape, dtype=bool)
    if add_cls:
        # cls位置上的所有特征维度，不遮盖
        mask[0] = True  # CLS at 0, set mask=1
    return mask

def geom_noise_mask_single(L, lm, masking_ratio):
    '''
    使用几何分布生成单个特征的掩码。
    Args:
        L: 轨迹长度, traj_cls_len
        lm: avg_mask_len, 2, 平均掩码长度，用于控制掩码的长度。
        masking_ratio: 掩码比例，表示需要掩码的数据比例。
    Returns:
        keep_mask: 生成的掩码，形状为 (traj_cls_len,)，True 表示保留，False 表示掩码。
    '''
    keep_mask = np.ones(L, dtype=bool)
    # 每个掩码序列停止的概率，几何分布的参数
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    # 每个未掩码序列停止的概率，几何分布的参数
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    # 概率列表，p[0] 是掩码状态的概率，p[1] 是未掩码状态的概率
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    # 初始状态，根据掩码比例决定是掩码状态还是未掩码状态
    # state 0 表示掩码，1 表示未掩码
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        # 根据当前状态设置掩码值
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        # 根据概率决定是否切换状态
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep_mask