# FFS-Mamba

The PyTorch implementation of my paper "Selective Fusion for Self-supervised Trajectory Representation Learning with Dual-View Mamba".

## Requirements
```bash
pytorch 2.5.1+cu124
mamba-ssm 2.2.6
```

## Pretrain

```bash
# 预训练
python ./ffs_pipeline_mlm_contra.py
```

## Downstream Tasks

### 1 Travel Time Estimation (TTE)

```bash
# TTE
# -- 整测 TERMba+SAGMba+CrossAttn
python ./ffs_pipeline_tte_crossattn.py
```

单测Stall GPS Mamba
Config：

- xian_tte
- chengdu_small_4_gpsview_in_tte_aug_20w

```bash
# -- 单测 Stall GPS Mamba
python ./ffs_pipeline_tte_aug.py
```


### 2 Next Location Prediction (NLP)

```bash
# Next Loc
# -- ⚠️ 提前需要摘取GpsRoadList
# -- Next Gps, 
python ./ffs_pipeline_dp_gps_origin.py
# -- Next Road
python ./ffs_pipeline_dp.py
```

### 3 Most Simliar Trajectory Search (MSTS)

```bash
# MSTS
python ./ffs_pipeline_msts.py
```

