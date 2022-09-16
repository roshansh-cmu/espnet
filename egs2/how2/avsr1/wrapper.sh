#! /bin/bash

gpu_id=$(($1-1))
echo $gpu_id
CUDA_VISIBLE_DEVICES=$gpu_id python local/prepare_visual_feats.py $1
