config = {

    "line":"ffs",
    # -- main args
    "task":"ffs_downstream",
    "model": "MambaMlmGpsRoadContra",
    "dataset": "chengdu",
    "config_file": "msts",
    "saved_model": True, # executor 参数
    "train": True,
    "device": None,

    # -- dp
    "dp_type": "gps", # gps or road
    # -- -- road executor/evaluator 需要使用 classifier
    # -- -- gps executor/evaluator 需要使用 regression
    "dp_len": 5, # 沿用TrajMamba中的设置，遮盖多少位置
    # dp gps时的部分
    "lng_max": 104.12907,
    "lng_min": 104.04211,
    "lat_max": 30.72649,
    "lat_min": 30.65283,

    # -- pretrain
    "pretrain_mamba_road_view_ckpt": "./libcity/cache/pm/star/936121/model_cache/936121_MambaRoadViewRoadGM_chengdu.pt",
    "pretrain_mamba_gps_view_ckpt": "./libcity/cache/ffs/249937/model_cache/MambaFuseAugView_chengdu_epoch8.tar",
    # "initial_ckpt": "./libcity/cache/ffs/843509/model_cache/843509_FFSTTE_AUG_chengdu.pt",
    "pretrain_mamba_ckpt": None,  # "./libcity/cache/ffs/124339/model_cache/MambaMlmGpsRoadContra_chengdu_epoch6.tar", # "./libcity/cache/ffs/74810/model_cache/MambaMlmGpsRoadContra_chengdu_epoch29.tar",

    # ConfigParser 赋值
    'exp_id': None,


    # -- vocab
    # -- -- index pad/cls/mask最好使用较小的数，避免nn.Embedding的时候需要重新映射
    "pad_index": 0, # padding_mask, padding_length for batch
    "unk_index": 1, #
    "sos_index": 2, # cls, start of sentence
    "mask_index": 3, # mask, span-mask, augument-mask


    # -- raw_data path
    "traj_path": "./raw_data/pm/chengdu/chengdu_trajs_2w.parquet",
    "road_path": "./raw_data/pm/chengdu/chengdu_roads.csv",
    "vocab_path": "./raw_data/pm/chengdu/chengdu_vocab.pkl",  # road_vocab
    "poi_path": "./raw_data/pm/chengdu/chengdu_pois.csv",
    "roadgat_neighbor_path": "./raw_data/pm/chengdu/chengdu_roadgat_neighbor.json",
    "roadgat_transprob_path": "./raw_data/pm/chengdu/chengdu_roadgat_transprob.json",
    "road_meta_path": "./raw_data/pm/chengdu/chengdu_roads_meta_selectWithDegree.csv",
    "rel_path": "./raw_data/pm/chengdu/chengdu_roads_rel_selectWithDegree.csv",
    # -- aug path:
    "most_sim_index_path": "/aoi/raw_data/20w/most_sim_index.npy",

    # -- cache path
    "use_cache": True,
    "cache_gps_traj_list_path": "/aoi/raw_data/20w/MambaFuseViewInnerDataset_GpsTrajList.pkl",
    "cache_road_traj_list_path": "/aoi/raw_data/20w/MambaFuseViewInnerDataset_RoadTrajList.pkl",
    "cache_road_traj_mat_list_path": "/aoi/raw_data/20w/MambaFuseViewInnerDataset_RoadTrajMatList.pkl",
    "cache_qry_gps_traj_list_path": "/aoi/raw_data/20w/most-sim/mostsim_gps_traj_list_qry.npy",
    "cache_tgt_gps_traj_list_path": "/aoi/raw_data/20w/most-sim/mostsim_gps_traj_list_tgt.npy",
    "cache_all_gps_traj_list_path": "/aoi/raw_data/20w/most-sim/mostsim_gps_traj_list_all.npy",
    "cache_neg_indices_path": "/aoi/raw_data/20w/most-sim/mostsim_neg_indices.npy",
    "cache_qry_tgt_indices_path": "/aoi/raw_data/20w/most-sim/mostsim_qry_tgt_indices.npy",

    # -- dataset
    "dataset_class": "FFS_Msts_Dataset",
    "batch_size": 64,
    "num_workers": 0,
    "seq_len": 128, # 不使用
    # -- -- road poi的表示，在训练得到(n,d)后 应用MeanPool降为(d), 不添加cls
    # -- -- -- pad_index和mask_index予以保留
    "add_cls_for_road": False, # 不使用cls，使用meanPool
    "add_cls_for_poi": False,
    "add_cls_for_gps": False,
    # -- dataset  -- span-mlm for road view traj
    "masking_ratio": 0.15,
    "avg_mask_len": 2,
    "masking_mode": "together",
    "distribution": "geometric",


    # -- model
    # -- pre
    "seed": 0,
    # -- For MambaRoadView
    # -- -- transformer_gat
    "transformer_d_model": 256, # 需要和mamba_gps_d_model保持一致才能继续训练。
    "transformer_n_layers": 6,
    "transformer_attn_heads": 8,
    "transformer_mlp_ratio": 4, # mlp中的ffn 的hidden size
    "transformer_dropout": 0.1,
    "drop_path": 0.3,
    "transformer_attn_drop": 0.1,
    "type_ln": "post", # 后 layernorm
    "future_mask": False,
    "gat_heads_per_layer": [8, 16, 1],
    "gat_features_per_layer": [16, 16, 256],
    "gat_dropout": 0.1,
    "gat_K": 1,
    "gat_avg_last": True,
    "add_minute_in_hour": True,
    "add_time_in_day": True,
    "add_day_in_week": True,
    "add_pe": True,
    "add_temporal_bias": True,
    "temporal_bias_dim": 64,
    "use_mins_interval": False,
    # -- RoadGM
    "roadgm_d_model": 256,  # 和transformer_d_model保持一致
    "roadgm_add_layer_norm": True,  # roadGM使用per node的layer norm比较合适
    "roadgm_add_batch_norm": False,
    "roadgm_gat_num_heads_per_layer": [8, 4, 4],
    "roadgm_gat_num_features_per_layer": [32, 64, 256],
    "roadgm_gat_bias": True,
    "roadgm_gat_dropout": 0.1,
    "roadgm_gat_avg_last": True,  # 以在最后一层使用4*64
    "roadgm_gat_load_trans_prob": True,
    "roadgm_gat_add_skip_connection": True,
    "roadgm_mamba_attn_dropout": 0.1,  #
    # -- For MambaRoadView
    "mamba_road_d_model": 256,
    "mamba_road_embed_size": 64,
    "mamba_road_use_mamba2": 1,
    "mamba_road_n_layer": 4,
    "mamba_road_d_state": 64,
    "mamba_road_head_dim": 64,
    "mamba_road_d_inner": 0,
    # -- For MambaGpsView
    "mamba_gps_d_model": 256,  # d_model
    "mamba_gps_embed_size": 64,  # 可学习参数的嵌入维度
    "use_mamba2": 2,  # 1 mamba, 2 mamba2, 0 TransformerEncoder
    "mamba_gps_n_layer": 4,  # mamba block 堆叠的层数
    "mamba_gps_d_state": 128,  # mamba blcok 的 state-size
    "mamba_gps_head_dim": 64,  # mamba block的head dimension
    "mamba_gps_d_inner": 0,  # inner model dimension of Traj-Mamba Blocks. If setting to 0 means d_inner=2*d_model.
    # -- Weight
    "road_weight": 1,
    "gps_weight": 1,



    # -- executor
    "executor": "FFSDP_Gps_Executor",
    "learner": "adamw",
    "learning_rate": 0.0002,
    "weight_decay": 0.01,
    "lr_beta1": 0.9,
    "lr_beta2": 0.999,
    "lr_alpha": 0.99,
    "lr_epsilon": 1e-8,
    "lr_momentum":0,
    "grad_accmu_steps": 1,
    "test_every": 1, # 多少个epoch evaluate测试一次

    "lr_decay": True,
    "lr_scheduler": "cosinelr",
    "lr_decay_ratio": 0.1,
    "steps": [],
    "step_size": 10,
    "lr_eta_min": 1e-5,
    "lr_patience": 10,
    "lr_threshold": 1e-4,
    "lr_warmup_epoch": 4,
    "lr_warmup_init": 1e-06,
    "t_in_epochs": True,

    "clip_grad_norm": True,
    "max_grad_norm": 5,
    "use_early_stop": True,
    "patience": 10,
    "log_every": 1,
    "log_batch": 250,
    "load_best_epoch": True,
    "l2_reg": None,

    "contra_loss_type": "simclr",
    "n_views": 2,
    "temperature": 0.05,
    "similarity": "cosine",

    # -- evaluator
    # -- -- gps FFSTTE_Evaluator ["MAE", "RMSE"] # , "MAPE", "R2", "EVAR"
    # -- -- road FFSDP_Evaluator ["Precision", "Recall", "F1", "MRR", "NDCG"]
    "evaluator": "FFSTTE_Evaluator",
    "metrics": ["MAE", "RMSE"], # , "MAPE", "R2", "EVAR"
    "save_modes": ["csv", "json"],
    "topk": [1, 5, 10],

}