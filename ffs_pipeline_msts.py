import numpy as np
from tqdm import tqdm
from einops import repeat, rearrange

from libcity.config import ConfigParser
from libcity.utils import get_executor, get_model, get_evaluator, get_dataset, \
    ensure_dir, set_random_seed, \
    get_logger, get_model_no_gat, get_evaluator_no_gat, get_executor_no_gat, cal_classification_metric, cal_mean_rank

config = ConfigParser(
    task="ffs_downstream",
    model="MambaMlmGpsRoadContra",
    dataset="chengdu",
    config_file="msts",
    saved_model=True,
    train=True,
    other_args=None
)
logger = get_logger(config, is_output_file=True)
logger.info('Begin pretrain-pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(config.get('task'), config.get("model"), config.get("dataset"), config.get("exp_id")))
logger.info("⚙️ Config Here...")
logger.info(config.config)
seed = config.get('seed', 0)
set_random_seed(seed)

dataset = get_dataset(config)
qry_data, tgt_data, all_data = dataset.get_data()
roadgat_data = dataset.get_roadgat_data()
graph_dict = {
    'node_features': roadgat_data.get("node_features"),
    'edge_index': roadgat_data.get("edge_index"),
    'edge_index_trans_prob': roadgat_data.get("edge_index_trans_prob"),
}
neg_indices = np.load(config.get("cache_neg_indices_path"), allow_pickle=True) # (num_tgt, num_negs)

model_cache_file = config.get('model_cache_file', './libcity/cache/{}/{}/model_cache/{}_{}_{}.pt'.format(
    config['line'], config['exp_id'], config['exp_id'], config['model'], config['dataset']))
model = get_model(config, roadgat_data)

model.to(config['device'])
model.eval()
# -- qry embedding
qry_embeds = []
for batch in tqdm(qry_data, desc='qry embedding', total=len(qry_data)):
    gps_X, gps_padding_mask, \
        road_X, road_padding_mask, road_traj_mat = batch
    embeds = model.forward_msts(gps_X, gps_padding_mask,
            road_X, road_padding_mask, road_traj_mat,
            graph_dict) # (B,D)
    qry_embeds.append(embeds.detach().cpu().numpy()) # len each (B,D)
qry_embeds = np.concatenate(qry_embeds, axis=0) # (num_tgt, D)
print("Debug here")

# -- tgt embedding
tgt_embeds = []
for batch in tqdm(tgt_data, desc='tgt embedding', total=len(tgt_data)):
    gps_X, gps_padding_mask, \
        road_X, road_padding_mask, road_traj_mat = batch
    embeds = model.forward_msts(gps_X, gps_padding_mask,
            road_X, road_padding_mask, road_traj_mat,
            graph_dict) # (B,D)
    tgt_embeds.append(embeds.detach().cpu().numpy()) # len each (B,D)
tgt_embeds = np.concatenate(tgt_embeds, axis=0) # (num_tgt, D)

# -- all embeds
all_embeds = []
for batch in tqdm(all_data, desc='all embedding', total=len(all_data)):
    gps_X, gps_padding_mask, \
        road_X, road_padding_mask, road_traj_mat = batch
    embeds = model.forward_msts(gps_X, gps_padding_mask,
            road_X, road_padding_mask, road_traj_mat,
            graph_dict) # (B,D)
    all_embeds.append(embeds.detach().cpu().numpy()) # len each (B,D)
all_embeds = np.concatenate(all_embeds, axis=0) # (num_traj_list, D)


def cal_pres_and_labels(query, target, negs):
    """
    query: (N, d)
    target: (N, d)
    negs: (N, n, d)
    """
    num_queries = query.shape[0]
    num_targets = target.shape[0]
    num_negs = negs.shape[1]
    print("query: ", query.shape)
    print("target: ", target.shape)
    print("neg: ", negs.shape)
    assert num_queries == num_targets, "Number of queries and targets should be the same."

    query_t = repeat(query, 'nq d -> nq nt d', nt=num_targets)
    query_n = repeat(query, 'nq d -> nq nn d', nn=num_negs)
    target = repeat(target, 'nt d -> nq nt d', nq=num_queries)
    # negs = repeat(negs, 'nn d -> nq nn d', nq=num_queries)

    dist_mat_qt = np.linalg.norm(query_t - target, ord=2, axis=2)
    dist_mat_qn = np.linalg.norm(query_n - negs, ord=2, axis=2)
    dist_mat = np.concatenate([dist_mat_qt[np.eye(num_queries).astype(bool)][:, None], dist_mat_qn], axis=1)

    pres = -1 * dist_mat
    labels = np.zeros(num_queries)

    return pres, labels

def cal_pres_and_labels_chunk(query, target, negs, chunk_size=512):
    """
    query:  (N, d)
    target: (N, d)
    negs:   (N, n, d)
    return:
        pres:   (N, 1+n)
        labels: (N,)
    """

    # ---------- 基本检查 ----------
    assert query.shape == target.shape
    N, d = query.shape
    n = negs.shape[1]
    num_chunks = (n + chunk_size - 1) // chunk_size

    # 强制 float32，减半内存
    query  = query.astype(np.float32)
    target = target.astype(np.float32)
    negs   = negs.astype(np.float32)

    # ---------- 正样本距离 ----------
    # (N,)
    pos_dist = np.linalg.norm(query - target, axis=1)

    # ---------- 负样本距离（分块） ----------
    neg_dist_chunks = []

    for i in tqdm(
        range(0, n, chunk_size),
        desc='Computing neg distances',
        total=num_chunks,
    ):
        # (N, chunk, d)
        neg_chunk = negs[:, i:i+chunk_size, :]

        # (N, chunk)
        dist = np.linalg.norm(
            query[:, None, :] - neg_chunk,
            axis=2
        )
        neg_dist_chunks.append(dist)

    # (N, n)
    neg_dist = np.concatenate(neg_dist_chunks, axis=1)

    # ---------- 拼接 ----------
    dist_mat = np.concatenate(
        [pos_dist[:, None], neg_dist],
        axis=1
    )

    pres = -dist_mat
    labels = np.zeros(N, dtype=np.int64)

    return pres, labels


predictions, targets = cal_pres_and_labels_chunk(qry_embeds, tgt_embeds, all_embeds[neg_indices])

metric = cal_classification_metric(targets, predictions)
metric["mean_rank"] = cal_mean_rank(predictions, targets)
print(f"the test metric for similar trajectory search:")
print(metric)


# executor = get_executor(config, model, roadgat_data)
#
# initial_ckpt = config.get("initial_ckpt", None)
# pretrain_path = config.get("pretrain_path", None)
# if config['train']:
#     executor.train(train_data, valid_data, test_data)
#     if config['saved_model']:
#         executor.save_model(model_cache_file)
#     executor.load_model(config.get("initial_ckpt"))
#     # executor.load_model_with_epoch(17)
#     # executor.evaluate(test_data)
# else:
#     # assert os.path.exists(model_cache_file) or initial_ckpt is not None or pretrain_path is not None
#     # if initial_ckpt is None and pretrain_path is None:
#     #     executor.load_model_state(model_cache_file)
#     if initial_ckpt is not None:
#         executor.load_model_with_tar(config.get("initial_ckpt"))
#     executor.evaluate(test_data)
#
#
