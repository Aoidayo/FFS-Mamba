import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from libcity.dataset.pm.road_vocab import RoadVocab
from libcity.utils import str2bool, ensure_dir

'''
使用时，需要修改
1、line、dataset
2、traj_path
3、将之前生成的vocab删除

生成
------
D:\code\simple\raw_data\pm\chengdu\cache\chengdu_roads_meta_select.csv
D:\code\simple\raw_data\pm\chengdu\cache\chengdu_roads_meta_selectWithDegree.csv
D:\code\simple\raw_data\pm\chengdu\cache\chengdu_roads_rel_select.csv
D:\code\simple\raw_data\pm\chengdu\cache\chengdu_roads_rel_selectWithDegree.csv
'''

line = 'pm'
dataset = "chengdu"
min_freq = 1
seq_len = 128

# -- 构建路径
road_path = f"./raw_data/{line}/{dataset}/{dataset}_roads.csv"
roads_meta_path = f"./raw_data/{line}/{dataset}/{dataset}_roads_meta.csv"
vocab_path = f"./raw_data/{line}/{dataset}/{dataset}_vocab.pkl"
# traj_path = f"./raw_data/{line}/{dataset}/{dataset}_trajs_1w.parquet"
traj_path = f"./raw_data/{line}/{dataset}/{dataset}_trajs_20w.parquetremove_last_timestamp.parquet"
rel_path = f"./raw_data/{line}/{dataset}/{dataset}_roads_rel.csv"

# -- 生成/加载vocab
if not os.path.exists(vocab_path):
    vocab = RoadVocab(traj_path=traj_path, min_freq=min_freq, use_mask=True, seq_len=seq_len)
    vocab.save_vocab(vocab_path)
else:
    vocab = RoadVocab.load_vocab(vocab_path)

# -- 选择min_freq >=1的road，生成cache下的roads.csv/roads_meta.csv/adjacency.npy
# -- -- Question:为什么要生成roads.csv?
# -- -- -- Answer: 生成roads.csv / rel.csv后 往其中插入 路网的出入度信息
# -- cache
cache_dir = f"./raw_data/{line}/{dataset}/cache/"
ensure_dir(cache_dir)
select_road_path = cache_dir + f"{dataset}_roads_meta_select.csv"

select_rel_path = cache_dir + f"{dataset}_roads_rel_select.csv"

road_ids = vocab.vocabIndex2roadIndexToken
# -- -- -- cache_dir + f"{dataset}_roads_meta_select.csv"
road_df = pd.read_csv(roads_meta_path, encoding='utf-8')
new_road_list = []
for i in tqdm(range(road_df.shape[0]), desc=f'处理 {select_road_path}'):
    # 将loc出来的单值 视作str
    road_id = int(road_df.loc[i, 'road_id'])
    if road_id in road_ids:
        new_road_list.append(road_df.iloc[i].values.tolist())
select_road_df = pd.DataFrame(new_road_list, columns=road_df.columns)
select_road_df.to_csv(select_road_path, index=False, encoding='utf-8')
print(f"✅{select_road_path}存储成功")

# -- -- -- cache_dir + f"{dataset}_roads_rel_select.csv"
rel_df = pd.read_csv(rel_path, encoding='utf-8')
new_rel_list = []
for i in tqdm(range(rel_df.shape[0]), desc=f'处理 {select_rel_path}'):
    src_id = int(rel_df.loc[i, 'src_id'])
    dst_id = int(rel_df.loc[i, 'dst_id'])
    if src_id not in road_ids or dst_id not in road_ids:
        continue
    new_rel_list.append(rel_df.iloc[i].values.tolist())
new_rel_df = pd.DataFrame(new_rel_list, columns=rel_df.columns)
new_rel_df.to_csv(select_rel_path, index=False, encoding='utf-8')
print(f"✅{select_rel_path}存储成功")


# -- 处理出入度
# -- -- 使用select_road_path, select_rel_path 生成出入度
# -- -- 构建路径
selectWithDegree_road_path = cache_dir + f"{dataset}_roads_meta_selectWithDegree.csv"
selectWithDegree_rel_path = cache_dir + f"{dataset}_roads_rel_selectWithDegree.csv"
road_ids = list(select_road_df['road_id'])
road2ind = {}
for i, road in enumerate(road_ids):
    road2ind[road] = i # road_id的行索引 i
adj_mx = np.zeros((len(road_ids), len(road_ids)), dtype=np.float32)
for row in tqdm(new_rel_df.values, desc=f'处理邻接矩阵'):
    if row[0] not in road2ind.keys() or row[1] not in road2ind.keys():
        print(row[0], row[1], "rel中的road，不在select的road中")
        continue
    # row[0], row[1] in road2ind.keys()
    adj_mx[road2ind[row[0]], road2ind[row[1]]] = 1 # (N,N)
outdegree = np.sum(adj_mx, axis=1)  # (N, )
indegree = np.sum(adj_mx.T, axis=1)  # (N, )
outdegree_list = []
indegree_list = []
for i, row in tqdm(select_road_df.iterrows(), total=select_road_df.shape[0], desc='in/out degree'):
    # road2ind[geo_id] = i
    geo_id = row['road_id']
    outdegree_i = outdegree[road2ind[geo_id]]
    indegree_i = indegree[road2ind[geo_id]]
    outdegree_list.append(int(outdegree_i))
    indegree_list.append(int(indegree_i))

select_road_df.insert(loc=select_road_df.shape[1], column='outdegree', value=outdegree_list)
select_road_df.insert(loc=select_road_df.shape[1], column='indegree', value=indegree_list)

new_rel_df.to_csv(selectWithDegree_rel_path, index=False)
select_road_df.to_csv(selectWithDegree_road_path, index=False)
print(f"✅{selectWithDegree_rel_path}存储成功")
print(f"✅{selectWithDegree_road_path}存储成功")