'''
根据所有轨迹生成 一次
'''
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from logging import getLogger
from libcity.utils.pm.read_parquet import ReadParquet
logger = getLogger(__name__)

def cal_matmul(mat1, mat2):
    n = mat1.shape[0]
    assert mat1.shape[0] == mat1.shape[1] == mat2.shape[0] == mat2.shape[1]
    res = np.zeros((n, n), dtype='bool')
    for i in tqdm(range(n), desc='outer'):
        for j in tqdm(range(n), desc='inner'):
            res[i, j] = np.dot(mat1[i, :], mat2[:, j])
    return res


city = "chengdu"
roads_meta_file = "./raw_data/pm/chengdu/chengdu_roads_meta.csv"
adjacency_file = "./raw_data/pm/chengdu/chengdu_adjacency.npy"
k_hop = 1
bidir_adj_mx = False
custom = True
transprob_mode = 'in'
neighbor_file = "./raw_data/pm/chengdu/chengdu_roadgat_neighbor.json"
transprob_path = "./raw_data/pm/chengdu/chengdu_roadgat_transprob.npy"
traj_file = "./raw_data/pm/chengdu/chengdu_trajs_20w.parquetremove_last_timestamp.parquet"


path_meta = pd.read_csv(roads_meta_file, encoding='utf-8')
adjacency = np.load(adjacency_file)
path_meta_ids = list(path_meta['road_id'])
path_meta_nums = len(path_meta_ids)

# [step neighbor：计算k_hop阶邻域，同时持久化为json文件]
if os.path.exists(neighbor_file):
    pathid2neighbors = json.load(open(neighbor_file, 'r'))
    logger.info("从{}中读取pathid2neighbors".format(neighbor_file))
else:
    adjacency = adjacency.T
    adj_mx = np.zeros((path_meta_nums, path_meta_nums), dtype=np.float32)
    for row in adjacency:
        adj_mx[row[0], row[1]] = 1
        if bidir_adj_mx:
            adj_mx[row[1], row[0]] = 1
    adj_mx_bool = adj_mx.astype('bool')
    k_adj_mx_list = [adj_mx_bool]
    # not in use
    for i in tqdm(range(2, k_hop + 1)):  # range(2,2) []
        # K = 2
        # [2,3), i=2
        if custom:
            k_adj_mx_list.append(cal_matmul(k_adj_mx_list[-1], adj_mx_bool))
        else:
            k_adj_mx_list.append(np.matmul(k_adj_mx_list[-1], adj_mx_bool))
        # np.save(os.path.join(base_path, '{0}/{0}_adj_{1}.npy'.format(road_name, i)), k_adj_mx_list[-1])
        # np.save( './raw_data/{0}/{0}_adj_{1}.npy'.format(city, i), k_adj_mx_list[-1])
    logger.info('Finish K order adj_mx')
    for i in tqdm(range(1, len(k_adj_mx_list))):
        adj_mx_bool += k_adj_mx_list[i]
    logger.info('Finish sum of K order adj_mx')
    # neighbors 计算
    pathid2neighbors = {}
    for i in tqdm(range(len(adj_mx_bool)), desc='count neighbors'):
        pathid2neighbors[i] = []
        for j in range(adj_mx_bool.shape[1]):
            if adj_mx_bool[i][j] == 0:
                continue
            pathid2neighbors[i].append(j)
    json.dump(pathid2neighbors, open(neighbor_file, 'w'))
    logger.info(f"已将pathid2neighbors持久化至{neighbor_file}中")
    logger.info('Total edge@{} = {}'.format(1, adj_mx.sum()))
    logger.info('Total edge@{} = {}'.format(k_hop, adj_mx_bool.sum()))

# [step transprob 计算hopk的转移概率]
if os.path.exists(transprob_path):
    transprob_matrix = np.load(transprob_path, allow_pickle=True)
    logger.info("从{}读取持久化的transprob_matrix".format(transprob_path))
else:
    node_array = np.zeros([path_meta_nums, path_meta_nums],dtype=float)
    count_array_row = np.zeros([path_meta_nums], dtype=int)
    count_array_col = np.zeros([path_meta_nums], dtype=int)
    train = ReadParquet(traj_file).read_parquet_with_timer()
    for _, row in tqdm(train.iterrows(), total=train.shape[0], desc='count traj prob'):
        cpath_list = row['road_list']
        for i in range(len(cpath_list)-1):
            # i \in [0, len(cpath_list)-2], 忽略最后一个len(cpath_list)-1
            prev_path_id = cpath_list[i]
            for j in range(1, k_hop+1):
                if i+j >= len(cpath_list):
                    continue
                next_path_id = cpath_list[i+j] # 注意，i+j才是下一个
                count_array_row[prev_path_id] += 1
                count_array_col[next_path_id] += 1
                node_array[prev_path_id][next_path_id] += 1
    # # node_array.sum(axis=1) 沿axis1求和，得到row； 应有如下逻辑
    assert ( count_array_row == (node_array.sum(axis=1)) ).sum() == len(count_array_row)  # 按行求和
    assert ( count_array_col == (node_array.sum(axis=0)) ).sum() == len(count_array_col)  # 按列求和

    # # [基于出度的转移概率]
    node_array_out = node_array.copy()
    no_outdegree_num = 0
    for i in tqdm(range(node_array_out.shape[0])):
        count = count_array_row[i]
        if count == 0:
            # logger.info(f'Node/Path {i} : no out-degree')
            no_outdegree_num += 1
            continue
        node_array_out[i, :] /= count
    logger.info(f'一共有 {no_outdegree_num}/{node_array_out.shape[0]} 条路段作为终点边，没有出度 ')

    # # [基于入度的转移概率]
    node_array_in = node_array.copy()
    no_indegree_num = 0
    for i in tqdm(range(node_array_in.shape[0])):
        count = count_array_col[i]
        if count == 0:
            # logger.info(f'Node/Path {i} : no in-degree')
            no_indegree_num += 1
            continue
        node_array_in[:, i] /= count
    logger.info(f'一共有 {no_indegree_num}/{node_array_in.shape[0]} 条路段作为起始边，没有入度 ')

    if transprob_mode == 'in':
        transprob_matrix = node_array_in.copy()
    elif transprob_mode == 'out':
        transprob_matrix = node_array_out.copy()
    else:
        raise AttributeError("transprob_mode must be in or out")

    np.save(transprob_path, transprob_matrix)
    logger.info("持久化transprob_matrix到{}".format(transprob_path))

logger.info("根据基于轨迹计算的transprob，给路网添加transprob")
# road_meta_transprob = "./raw_data/{0}/{0}_transprob_{2}_hop{1}.json".format(city,k_hop,transprob_mode)
road_meta_transprob = "./raw_data/pm/chengdu/chengdu_roadgat_transprob.json"
if os.path.exists(road_meta_transprob):
    logger.info("从{}读取transprob.json".format(road_meta_transprob))
    neighbor_file = json.load(open(road_meta_transprob,'r'))
else:
    neighbor2prob = {}
    for k,v in pathid2neighbors.items():
        for tgt in v:
            id_ = "{}_{}".format(int(k),int(tgt))
            p_ = transprob_matrix[int(k)][int(tgt)]
            neighbor2prob[id_] = float(p_)
    json.dump(neighbor2prob,open(road_meta_transprob,'w'))
    logger.info("持久化tranprob的json格式文件到{}".format(road_meta_transprob))