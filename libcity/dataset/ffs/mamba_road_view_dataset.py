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
import datetime

class MambaRoadViewDataset(object):
    def __init__(self, config):

        self._logger = getLogger()
        self.config = config

        self.line = self.config.get('line', 'pm')
        self.dataset = self.config.get('dataset', 'chengdu')
        self.vocab_path = self.config.get('vocab_path', None)
        self.traj_path = self.config.get("traj_path", None)
        self.traj_train_path = self.traj_path[:-8]+"_train.parquet"
        self.traj_eval_path = self.traj_path[:-8]+"_eval.parquet"
        self.traj_test_path = self.traj_path[:-8]+"_test.parquet"
        self.road_path = self.config.get('road_path', None)
        self.road_meta_path = self.config.get('road_meta_path', None)
        self.rel_path = self.config.get('rel_path', None)
        self.cache_path = f"./raw_data/{self.line}/{self.dataset}/cache/"
        ensure_dir(self.cache_path)

        self.batch_size = config.get('batch_size', 64)
        self.seq_len = self.config.get('seq_len', 64)
        self.num_workers = self.config.get('num_workers', 0)
        self.add_cls_for_road = config.get('add_cls_for_road', False)
        self.add_cls_for_poi = config.get('add_cls_for_poi', False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -- 加载road vocab
        self.driver_num = 0
        self.vocab_size = 0
        self.vocab = None
        self.__load_vocab()

        # -- 准备和roadGat相关的内容
        self.road_df = None
        self.rel_df = None
        self.road_size = None
        self.node_features = None # (vocab_size, node_feature_dim)
        self.node_fea_dim = 0 # 路段维度
        self.edge_index = None # 边索引 (2, E), rel_df的转置
        self.edge_index_trans_prob = None # 路段转移索引
        self.roadgat_neighbor_path = self.config.get('roadgat_neighbor_path', None)
        self.roadgat_transprob_path = self.config.get('roadgat_transprob_path', None)
        self.__prepare_roadgat()

        # -- span-mlm
        self.masking_ratio = self.config.get('masking_ratio', 0.15)
        self.avg_mask_len = self.config.get('avg_mask_len', 2)
        self.masking_mode = self.config.get('masking_mode', 'together')
        self.distribution = self.config.get('distribution', None)

        # -- dataset 2 dataloader
        self.collate_fn = PretrainCollateFnClass(max_len=self.seq_len, vocab=self.vocab, add_cls_for_road=self.add_cls_for_road, add_cls_for_poi=self.add_cls_for_poi)

        # -- dataset --shuffle--> dataloader, 默认True，shuffle Train和Test
        self.dataloader_shuffle = self.config.get("dataloader_shuffle", True)


    def __load_vocab(self):
        '''
        - 使用RoadVocab加载预构建的vocab
        - 初始化usrnum、vocab_size
        Returns:

        '''
        self._logger.info("Loading Vocab from {}".format(self.vocab_path))
        self.vocab = RoadVocab.load_vocab(self.vocab_path)
        self.driver_num = self.vocab.driver_num
        self.vocab_size = len(self.vocab)
        self._logger.info('vocab_path={}, driver_num={}, vocab_size={}'.format(
            self.vocab_path, self.driver_num, self.vocab_size))


    def __prepare_roadgat(self):
        self.road_df = pd.read_csv(self.road_meta_path, encoding='utf-8')
        self.rel_df = pd.read_csv(self.rel_path, encoding='utf-8')
        self.road_id_list = list(self.road_df['road_id'])
        self.road_size = len(self.road_id_list)

        """
        -- vocab相关: 
        'node_features' (vocab_size, road_dim)
        -- 与vocab无关
        'edge_index'
        'edge_index_trans_prob'
        """

        # -- 1 \ 处理node_features
        # -- node_featuers: 处理归一化的路段特征 为 npy，shape:(vocab_size, node_feature_dim)
        node_features_path = self.cache_path + f"{self.dataset}_node_features_degree.npy"
        if os.path.exists(node_features_path):
            node_features = np.load(node_features_path, allow_pickle=True)
        else:
            '''
            highway_id: 路段类型，已经二值化过
            length_id: 路段长度，分级处理过（比如30m以下的就是0, 2000m以上的是13，可以直接用于使用，不需要二次归一化处理）
            
            -- 不使用
            lanes: 有缺失值，无法直接补全，不使用
            '''
            useful_column = ['oneway', 'highway_id', 'length_id', 'road_speed', 'traj_speed', 'outdegree', 'indegree']
            node_features = self.road_df[useful_column]
            # 归一
            norm_column = [ 'traj_speed', 'outdegree']
            norm_dict = { column: idx for idx, column in enumerate(useful_column) if column in norm_column }
            for column, idx in norm_dict.items():
                ser = node_features[column]
                min_ = ser.min()
                max_ = ser.max()
                dnew = ( ser - min_ ) / ( max_ - min_ )
                # TODO: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html#pandas.DataFrame.drop
                # [NOTE]: pd.DataFrame.drop(labels=...,axis=1)
                node_features = node_features.drop(column, axis=1)
                node_features.insert(idx, column, dnew)
            onehot_column = [ 'oneway', 'highway_id', 'length_id','outdegree', 'indegree']
            for column in onehot_column:
                # TODO note: pd.get_dummies(pd.Series(...)) https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
                # pd.get_dummies(dataset:pd.Series, prefix:str)
                # 表示在生成的哑变量列名前添加的前缀。例如，如果 col 是 "color"，并且这一列中有 "red" 和 "blue" 两个类别，
                # 生成的列名可能是 "color_red" 和 "color_blue"。
                dum_col = pd.get_dummies(node_features[column], column) # len_road, one_hot_size
                node_features = node_features.drop(column, axis=1)
                node_features = pd.concat([node_features, dum_col], axis=1)
            node_features = node_features.values # (len_road, process_feature_dim)
            np.save(node_features_path, node_features)
        self._logger.info("node_features:" + str(node_features.shape))

        # node_feature -> vocab_node_feature
        # vocab_size, process_feature_dim
        node_fea_vec = np.zeros((self.vocab.vocab_size, node_features.shape[1])) # (vocab_size, fea_dim)
        for ind in range(len(node_features)):
            # self._load_geo中创建的，index和geoId互换的dict
            road_id = self.road_df.loc[ind, 'road_id']
            vocab_road_id = self.vocab.roadIndex2vocabIndexDict[road_id]
            node_fea_vec[vocab_road_id] = node_features[ind]
        node_fea_pe = torch.from_numpy(node_fea_vec).float().to(self.device)
        self._logger.info('vocab_node_features_encoded: ' + str(node_fea_pe.shape))
        self.node_features = node_fea_pe  # (vocab_size,fea_dim), fea_dim
        self.node_fea_dim = node_fea_pe.shape[1]


        # -- 2 \ 处理edge_index: (2,E);
        #           edge_index_trans_prob: E
        # -- -- 这里构建edge_index的方法非常简单，就是使用过滤后的chengdu_roads_rel_selectWithDegree.csv构建路网
        # { k:str = 'roadId',v:list = [roadId1, roadId2, ...] }
        roadId_to_neighbors = json.load(open(self.roadgat_neighbor_path, 'r'))
        # { k:str = 'roadId_roadId', v:float 转移概率 }
        roadedge_tranprob = json.load(open(self.roadgat_transprob_path, 'r'))
        source_nodes_ids, target_nodes_ids = [], [] # 存储边的两端road
        seen_edges = set()
        road_tranprob = [] # 存储边的转移概率

        for src_roadId, neighborRoadId in roadId_to_neighbors.items():
            '''
                vocab <-  traj 构建
                neighbor <- road_rel 构建
                trans_prob <- 基于neighbor，在traj上计算
                
                vocab中存在的road，肯定可以对应到neighbor中
                但是neighbor中存在的roadId，vocab中就不一定有
            '''
            # 如果neighbor中的roadId，在vocab中没有，直接跳过
            if int(src_roadId) not in self.vocab.vocabIndex2roadIndexToken:
                continue
            # roadId : str
            src_vocabId = self.vocab.roadIndex2vocabIndexDict[int(src_roadId)] # vocabIndex node
            for tgt_roadId in neighborRoadId:
                # 如果neighbor中的roadId，在vocab中没有，直接跳过
                if tgt_roadId not in self.vocab.vocabIndex2roadIndexToken:
                    continue
                tgt_vocabId = self.vocab.roadIndex2vocabIndexDict[int(tgt_roadId)]
                if (src_vocabId, tgt_vocabId) not in seen_edges:
                    # 添加有向边, 及其转移概率
                    source_nodes_ids.append(src_vocabId)
                    target_nodes_ids.append(tgt_vocabId)
                    seen_edges.add((src_vocabId, tgt_vocabId))
                    road_tranprob.append(roadedge_tranprob[str(src_roadId) + '_' + str(tgt_roadId)])

        # 添加包括 token在内的所有vocab_road的转移概率
        for i in range(self.vocab.vocab_size):
            if (i, i) not in seen_edges:
                source_nodes_ids.append(i)
                target_nodes_ids.append(i)
                seen_edges.add((i, i))
                road_tranprob.append( roadedge_tranprob.get(str(i) + '_' + str(i), 0.0) )
                if (road_tranprob[-1] != 0.0):
                    self.logger.warning("source_node:{},target_node:{} 存在自边".format(i, i))
                    raise RuntimeError("error!")

        # (2, E), edge_index[0,:] src_vocabId, edge_index[1,:] tgt_vocabId
        self.edge_index = torch.from_numpy(np.row_stack((source_nodes_ids, target_nodes_ids))).long().to(self.device)
        # (E) -unsqueeze(1)-> (E,1)
        # edge_index_trans_prob[index][0] ： 对应edge_index[:,index]的转移概率
        self.edge_index_trans_prob = torch.from_numpy(np.array(road_tranprob)).unsqueeze(1).float().to(self.device)

    def get_roadgat_data(self):
        return {
            "driver_num": self.driver_num,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "road_size": self.road_size,
            "road_df": self.road_df,
            "rel_df": self.rel_df,
            # (vocab_size, fea_dim)
            "node_features": self.node_features,
            "node_fea_dim": self.node_fea_dim,
            # 边索引，(2,E), 索引是roadVocabIndex
            "edge_index": self.edge_index,
            # 边索引 对于的转移概率 (E,1)
            "edge_index_trans_prob": self.edge_index_trans_prob,
        }

    def get_data(self):
        # Usage: 供外部调用，直接获取dataloader
        self._logger.info("生成 Dataset!")
        train_dataset, eval_dataset, test_dataset = self.__gen_dataset()
        self._logger.info('Size of dataset[Train Eval Test]: ' +
                         str(len(train_dataset)) + '/' + str(len(eval_dataset)) + '/' + str(len(test_dataset)))
        self._logger.info("生成 Dataloader!")
        return self.__gen_dataloader(train_dataset, eval_dataset, test_dataset)

    def __gen_dataset(self):
        train_dataset = TrajBaseDataset(
            traj_data_path=self.traj_train_path,
            vocab = self.vocab,
            cache_path=self.cache_path,
            config = self.config,
            type='train',
        )
        eval_dataset = TrajBaseDataset(
            traj_data_path=self.traj_eval_path,
            vocab = self.vocab,
            cache_path=self.cache_path,
            config=self.config,
            type='eval',
        )
        test_dataset = TrajBaseDataset(
            traj_data_path=self.traj_test_path,
            vocab = self.vocab,
            cache_path=self.cache_path,
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
                                      shuffle= True if self.dataloader_shuffle else False ,
                                      collate_fn=lambda raw_batch: self.collate_fn(raw_batch))
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     shuffle= True if self.dataloader_shuffle else False,
                                     collate_fn=lambda raw_batch: self.collate_fn(raw_batch))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     shuffle=False,
                                     collate_fn=lambda raw_batch: self.collate_fn(raw_batch))
        return train_dataloader, eval_dataloader, test_dataloader




class TrajBaseDataset(Dataset):
    def __init__(self, traj_data_path, vocab, cache_path, config, type):
        '''
        Args:
        ------
        traj_data_path:
            eg.sample dataset: D:\code\simple\raw_data\pm\chengdu\chengdu_trajs_1w.parquet
        cache_path:
            raw_data\pm\chengdu\cache
            生成如下几个
                gps_traj_list.npy
                road_traj_list.npy
                # gps_traj_mat.npy
                road_traj_mat.npy

        '''
        self._logger = getLogger()

        self.traj_data_path = traj_data_path
        self.vocab = vocab
        self.cache_path = cache_path
        self.gps_traj_list_path = self.cache_path + f"gps_traj_list_{type}.pkl"
        self.road_traj_list_path = self.cache_path + f"road_traj_list_{type}.pkl"
        self.road_traj_mat_path = self.cache_path + f"road_traj_mat_{type}.pkl"

        self.use_cache = config.get("use_cache",False)
        self.cache_gps_traj_list_path = config.get("cache_gps_traj_list_path")
        self.cache_road_traj_list_path = config.get("cache_road_traj_list_path")
        self.cache_road_traj_mat_list_path = config.get("cache_road_traj_mat_list_path")
        self.cache_gps_traj_list_path = self.cache_gps_traj_list_path[:-4] + f"_{type}.pkl"
        self.cache_road_traj_list_path = self.cache_road_traj_list_path[:-4] + f"_{type}.pkl"
        self.cache_road_traj_mat_list_path = self.cache_road_traj_mat_list_path[:-4] + f"_{type}.pkl"

        self._load_data()

        # -- span-mlm
        self.config = config
        self.masking_ratio = self.config.get('masking_ratio', 0.15)
        self.avg_mask_len = self.config.get('avg_mask_len', 2)
        self.masking_mode = self.config.get('masking_mode', 'together')
        self.distribution = self.config.get('distribution', None)


    def _load_data(self):

        if self.use_cache:
            self._logger.info("🤗 Use Cache 20w here")
            self.gps_traj_list = pickle.load(open(self.cache_gps_traj_list_path, 'rb'))
            self._logger.info(f"🤗 Use Cache 20w here: gps_traj_list loaded from {self.cache_gps_traj_list_path}")
            self.road_traj_list = pickle.load(open(self.cache_road_traj_list_path, 'rb'))
            self._logger.info(f"🤗 Use Cache 20w here: road_traj_list loaded from {self.cache_road_traj_list_path}")
            self.road_traj_mat_list = pickle.load(open(self.cache_road_traj_mat_list_path, 'rb'))
            self._logger.info(f"🤗 Use Cache 20w here: road_traj_mat_list loaded from {self.cache_road_traj_mat_list_path}")
        else:
            if os.path.exists(self.gps_traj_list_path) and os.path.exists(self.road_traj_list_path) \
                and os.path.exists(self.road_traj_mat_path):
                self.gps_traj_list = pickle.load(open(self.gps_traj_list_path,'rb'))
                self.road_traj_list = pickle.load(open(self.road_traj_list_path, 'rb'))
                self.road_traj_mat_list = pickle.load(open(self.road_traj_mat_path, 'rb'))
            else:
                self.gps_traj_list, self.road_traj_list, self.road_traj_mat_list = self.data_processing()
                pickle.dump(self.gps_traj_list, open(self.gps_traj_list_path, 'wb'))
                pickle.dump(self.road_traj_list, open(self.road_traj_list_path, 'wb'))
                pickle.dump(self.road_traj_mat_list, open(self.road_traj_mat_path, 'wb'))

    def __getitem__(self, index):
        gps_traj = self.gps_traj_list[index] # nd (len_gps_traj, f_traj)
        road_traj = self.road_traj_list[index] # (len_road_traj, f_road)
        road_traj_mat = self.road_traj_mat_list[index] # (len_road_traj, len_road_traj)
        # (len_road_traj, f-road)
        road_span_mlm_mask = noise_mask_last_pred_len(road_traj, 1)
        return gps_traj, road_traj, road_traj_mat, road_span_mlm_mask

    def __len__(self):
        return len(self.gps_traj_list)

    def data_processing(self):
        '''
        将所有特征 处理为 float Ndarray

        Returns:
            gps_traj_list: gps_traj(len_gps_traj, f_gps)
            road_traj_list: road_traj(len_road_traj, f_road)
            len_dataset: dataset中所有轨迹的数量

        '''
        # [len_df, ...]
        origin_df = pd.read_parquet(self.traj_data_path, engine='fastparquet')
        '''
        [
         -- traj's
         'traj_id', , 'traj_length', 'start_time', 'total_time', 'road_nums', 'gps_nums', 'driver_id'
         -- road's
         'road_list', 'road_tm_list', 'road_interval',
         -- gps's
         'gps_road_list', 'gps_tm_list', 'gps_speed_list', 'gps_acceleration_list', 'gps_angle_delta_list', 
         'gps_interval_list', 'gps_dist_list', 'gps_lat_list', 'gps_lng_list', 
        ]
        '''
        # -- gps
        # sub_df = origin_df[['gps_road_list', 'gps_tm_list', 'gps_lng_list', 'gps_lat_list', 'start_time', 'driver_id', 'traj_id']]
        gps_traj_list =[] # len_df
        road_traj_list = [] # len_df
        road_traj_mat_list = [] # len_df
        for i in tqdm(range(math.floor(origin_df.shape[0])),desc='处理 gps_traj, road_traj, road_traj_mat'):
            traj = origin_df.iloc[i]

            # -- gps
            gps_road_list = [ self.vocab.roadIndex2vocabIndexDict.get(road, self.vocab.unk_index) for road in traj['gps_road_list']]
            gps_tm_list = traj['gps_tm_list']
            driver_id = self.vocab.driverIndex2vocabIndexDict[traj['driver_id']]
            new_gps_tm_list = [ pd.to_datetime(tm, unit='s')  for tm in gps_tm_list ]
            gps_lng_list = traj['gps_lng_list']
            gps_lat_list = traj['gps_lat_list']
            start_time = pd.to_datetime(traj['start_time'], unit='s')
            gps_delta_time_list = [ (tm - start_time).total_seconds() for tm in new_gps_tm_list] #
            # -- gps traj numpy: (traj_len, f=[road, tm, lng, lat, delta_time ])
            # f_len * traj_len -> traj_len , f_len
            traj_fea = np.array([gps_road_list, gps_tm_list, gps_lng_list, gps_lat_list, gps_delta_time_list]).transpose((1,0))
            gps_traj_list.append(traj_fea)

            # -- road
            road_list = [ self.vocab.roadIndex2vocabIndexDict[road] for road in traj['road_list'] ]
            road_tm_list = traj['road_tm_list']
            new_road_tm_list = [ pd.to_datetime(tm, unit='s')  for tm in road_tm_list ]
            road_minutes_list = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_road_tm_list] # minute 0~59, hour 0~23
            road_weeks_list = [new_tim.weekday() + 1 for new_tim in new_road_tm_list] # weekday (0, 6)
            driver_list = [ driver_id ] * len(road_list)
            road_traj_fea = np.array([road_list, road_tm_list, road_minutes_list, road_weeks_list, driver_list ]).transpose((1,0))
            road_traj_list.append(road_traj_fea)

            # -- road matrix
            road_traj_mat = self.__cal_mat(road_tm_list)
            road_traj_mat_list.append(road_traj_mat)

        return gps_traj_list, road_traj_list, road_traj_mat_list

    def __cal_mat(self, tim_list):
        '''
        calculate the temporal relation matrix

        Args:
            tim_list: （traj_len)

        Returns:
            ndarray: mat[i][j] = abs( tim_list[i] - tim_list[j] )
                (traj_len,traj_len)
        '''
        # calculate the temporal relation matrix
        seq_len = len(tim_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                off = abs(tim_list[i] - tim_list[j])
                mat[i][j] = off
        return mat  # (seq_len, seq_len)


class PretrainCollateFnClass:
    def __init__(self, max_len, vocab, add_cls_for_road,  add_cls_for_poi):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len # 不使用max_len, 让gps和road X的最长长度决定
        self.vocab = vocab
        self.add_cls_for_road = add_cls_for_road # 不使用cls，使用meanPool
        self.add_cls_for_poi = add_cls_for_poi

    def __call__(self, raw_batch):
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

        gps_traj_batch, road_traj_batch, road_traj_mat_batch, road_span_mlm_mask_batch = [], [], [], []
        for row in raw_batch:
            gps_traj_batch.append(row[0])
            road_traj_batch.append(row[1])
            road_traj_mat_batch.append(row[2])
            road_span_mlm_mask_batch.append(row[3])
        batch_size = len(gps_traj_batch)
        gps_valid_lengths = [ x.shape[0] for x in gps_traj_batch ]
        road_valid_lengths = [ x.shape[0] for x in road_traj_batch ]
        max_gps_len = max(gps_valid_lengths)
        max_road_len = max(road_valid_lengths)

        gps_X = torch.zeros(batch_size, max_gps_len, gps_traj_batch[0].shape[-1], dtype=torch.float32)
        road_X = torch.zeros(batch_size, max_road_len, road_traj_batch[0].shape[-1], dtype=torch.float32)
        road_X.fill_(self.vocab.pad_index) # 默认用0遮盖
        road_target_masks = torch.zeros_like(road_X, dtype=torch.bool)
        road_traj_mat = torch.zeros(batch_size, max_road_len, max_road_len, dtype=torch.float32)
        for i in range(batch_size):
            gps_end = min(gps_valid_lengths[i], max_gps_len)
            # (gps_traj_len, f)
            gps_X[i, :gps_end, :] = torch.tensor(gps_traj_batch[i][:gps_end, :], dtype=torch.float32)

            road_end = min(road_valid_lengths[i], max_road_len)
            road_X[i, :road_end, :] = torch.tensor(road_traj_batch[i][:road_end, :], dtype=torch.float32)
            road_target_masks[i, :road_end, :] = torch.tensor(road_span_mlm_mask_batch[i][:road_end, :], dtype=torch.bool)
            road_traj_mat[i, :road_end, :road_end] = torch.tensor(road_traj_mat_batch[i][:road_end, :road_end], dtype=torch.float32)


        # (batch_size, len_max_gps)
        gps_padding_mask = padding_mask(torch.tensor(gps_valid_lengths, dtype=torch.int16), max_len=max_gps_len)
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
        gps_X.masked_fill_(gps_padding_mask.unsqueeze(-1) ==0, self.vocab.pad_index) # 默认 gpsX的所有特征 都用0遮盖
        # road_X[..., 0:1]: (batch_size, len_max_road, 1), f_road的第一个就是路段特征
        # road_X[..., 0]: (batch_size, len_max_road), ❌
        # 可见：test.mytorch.torchapis.testDotDotDot
        road_X[..., 0:1].masked_fill_(road_target_masks[...,0:1] == 1, self.vocab.mask_index) # mask
        road_X[...,1:].masked_fill_(road_target_masks[...,1:] == 1, self.vocab.pad_index)

        # gps_traj_batch = torch.from_numpy(pad_batch(gps_traj_batch)).float().to(self.device)
        # road_traj_batch = torch.from_numpy(pad_batch(road_traj_batch)).float().to(self.device)



        return (
            # -- gps
            gps_X.to(self.device), # (batch_size, len_max_gps, f_gps)
            gps_padding_mask.to(self.device),  # (batch_size, len_max_gps, len_max_gps)
            # -- road
            road_X.to(self.device), # (batch_size, len_max_road, f_road) maskIndex遮盖vocab_road_id，padIndex遮盖特征
            road_Target.to(self.device), # (batch_size, len_max_road, f_road) 遮盖的vocab_road_id和feature真值, 其余为pad_index
            road_padding_mask.to(self.device), # (batch_size, len_max_road)
            road_target_masks.to(self.device), #  (batch_size, len_max_road, f_road) True遮盖，False保留
            road_traj_mat.to(self.device), # (batch_size, len_max_road, len_max_road)
        )




'''
Helper Function:
 - padding_mask
 - nosie_mask
    - geom_noise_mask_single
'''

def noise_mask_last_pred_len(X, pred_len=5):
    '''
    False遮盖, True保留

    Parameters
    ----------
    X : (traj_cls_len, feat_dim)
    pred_len : 5

    Returns
    -------
    mask: ndarray(traj_cls_len, feat_dim)

    '''
    mask = np.ones(X.shape, dtype=bool) # True全局保留
    seq_len = X.shape[0]
    assert pred_len < seq_len, "pred_len must < seq_len"
    mask[-pred_len:, :] = False # 遮盖
    return mask

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
    L, F = X.shape
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
    elif distribution == 'interval':
        # add_cls=True 时，对 [1:] 做间隔遮盖，保证 CLS 不参与比例计算
        L_eff = L - 1  # if (add_cls and L > 0) else L

        def gen_one_col():
            col = np.ones(L, dtype=bool)
            if L_eff > 0:
                core = _interval_mask_single(L_eff, masking_ratio, jitter=True, exact=True)
                col[1:] = core
                # if add_cls and L > 0:
                #     col[1:] = core
                # else:
                #     col[:] = core
            return col

        if mode == 'separate':
            mask = np.ones(X.shape, dtype=bool)
            for m in range(F):
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = gen_one_col()
        else:
            one_col = gen_one_col()[:, None]
            mask = repeat(one_col, "traj_cls_len 1 -> traj_cls_len feature_dim", feature_dim=F)
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

def _interval_mask_single(L: int, masking_ratio: float, jitter: bool = True, exact: bool = True) -> np.ndarray:
    """
    间隔遮盖（单列）
    True=保留, False=遮盖
    masking_ratio=0.25 => 约每4个遮1个
    """
    keep = np.ones(L, dtype=bool)
    if L <= 0:
        return keep

    r = float(masking_ratio)
    r = min(max(r, 0.0), 1.0)
    if r <= 0:
        return keep
    if r >= 1:
        return np.zeros(L, dtype=bool)

    if exact:
        target = int(np.round(r * L))
        target = np.clip(target, 0, L)
        if target == 0:
            return keep
        if target == L:
            return np.zeros(L, dtype=bool)

        # 均匀撒点：floor(offset + i * L/target)
        offset = np.random.rand() * (L / target) if jitter else 0.0
        idx = np.floor(offset + np.arange(target) * (L / target)).astype(int)
        idx = np.clip(idx, 0, L - 1)

        idx = np.unique(idx)
        # 去重后可能少于 target，补齐
        if idx.size < target:
            cand = np.setdiff1d(np.arange(L), idx, assume_unique=False)
            need = target - idx.size
            extra = np.random.choice(cand, size=need, replace=False) if need > 0 else np.array([], dtype=int)
            idx = np.sort(np.concatenate([idx, extra]))

        keep[idx] = False
        return keep

    # 非 exact：误差扩散式（也很均匀）
    acc = np.random.rand() if jitter else 0.0
    for i in range(L):
        acc += r
        if acc >= 1.0:
            keep[i] = False
            acc -= 1.0
    return keep
