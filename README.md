# FFS-Mamba

The PyTorch implementation of my paper "Selective Fusion for Self-supervised Trajectory Representation Learning with Dual-View Mamba".

## Note 
The repository is currently being organized and will be gradually refined after the paper is submitted.

## Requirements
```bash
pytorch 2.5.1+cu124
mamba-ssm 2.2.6
```

## Pretrain

```bash
python ./ffs_pipeline_mlm_contra.py
```

## Downstream Tasks

### 1 Travel Time Estimation (TTE)

```bash
# TTE
python ./ffs_pipeline_tte_crossattn.py 
# or
python ./ffs_pipeline_tte_aug.py
```


### 2 Destionation Prediction

Mask 10% tail GPS and corresponding road, use the model to predict the destination road ID.

```bash
python ./ffs_pipeline_dp.py
```

### 3 Most Simliar Trajectory Search (MSTS)

```bash
python ./ffs_pipeline_msts.py
```

