'''
同时使用 GPS-Contra和Road-GPS Match
消融后发现，GPS-Contra Match是最有用的，消融此部分再做测试。
'''
config = {

    "line":"ffs",
    # -- main args
    "task":"ffs_pretrain",
    "model": "MambaFuseAugView",
    "dataset": "chengdu",
    "config_file": "chengdu_small_aug_20w_gc",
    "saved_model": True, # executor 参数
    "train": True,
    "device": None,

    # -- pretrain
    "pretrain_mamba_road_view_ckpt": "./libcity/cache/pm/763276/model_cache/MambaRoadViewRoadGM_chengdu_epoch46.tar",
    "pretrain_mamba_gps_view_ckpt": "./MambaGpsView_chengdu.tar",


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


    # -- dataset
    "dataset_class": "MambaFuseAugViewDataset",
    "batch_size": 32,
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
    # -- add for lng lat normalize & road -> vocab_road
    "denormalize": False,
    "lng_max": 104.12908,
    "lng_min": 104.04210,
    "lat_max": 30.72650,
    "lat_min": 30.65282,
    "vocab_size": 6253, # from chengdu_vocab


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
    "roadgm_d_model": 256, # 和transformer_d_model保持一致
    "roadgm_add_layer_norm": True, # roadGM使用per node的layer norm比较合适
    "roadgm_add_batch_norm": False,
    "roadgm_gat_num_heads_per_layer": [8, 4, 4],
    "roadgm_gat_num_features_per_layer": [32, 64, 256],
    "roadgm_gat_bias": True,
    "roadgm_gat_dropout": 0.1,
    "roadgm_gat_avg_last": True, # 以在最后一层使用4*64
    "roadgm_gat_load_trans_prob": True,
    "roadgm_gat_add_skip_connection": True,
    "roadgm_mamba_attn_dropout": 0.1, #
    # -- For MambaGpsView
    "mamba_gps_d_model": 256,  # d_model
    "mamba_gps_embed_size": 64,  # 可学习参数的嵌入维度
    "use_mamba2": 2,  # 1 mamba, 2 mamba2, 0 TransformerEncoder
    "mamba_gps_n_layer": 4,  # mamba block 堆叠的层数
    "mamba_gps_d_state": 128,  # mamba blcok 的 state-size
    "mamba_gps_head_dim": 64,  # mamba block的head dimension
    "mamba_gps_d_inner": 0,  # inner model dimension of Traj-Mamba Blocks. If setting to 0 means d_inner=2*d_model.
    # -- Weight
    "road_weight": 0.2,
    "gps_weight": 0.8,



    # -- executor
    "executor": "MambaFuseAugViewExecutorGC",
    "learner": "adamw",
    "learning_rate": 0.0002,
    "weight_decay": 0.01,
    "lr_beta1": 0.9,
    "lr_beta2": 0.999,
    "lr_alpha": 0.99,
    "lr_epsilon": 1e-8,
    "lr_momentum":0,
    "grad_accmu_steps": 1,
    "test_every": 10, # 多少个epoch evaluate测试一次

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
    "log_batch": 500,
    "load_best_epoch": True,
    "l2_reg": None,

    "contra_loss_type": "simclr",
    "n_views": 2,
    "temperature": 0.05,
    "similarity": "cosine",
    'align_w': 1.0,
    'unif_w': 1.0,
    'align_alpha': 2,
    'unif_t': 2,
    'train_align_uniform': False,
    'test_align_uniform': True,
    'norm_align_uniform': False,
    'rm_ratio': 0.3,
    'gc_ratio': 0.7,

    # -- evaluator
    "evaluator": None,
    "metrics": ["Precision", "Recall", "F1", "MRR", "NDCG"],
    "save_modes": ["csv", "json"],
    "topk": [1, 5, 10],

}