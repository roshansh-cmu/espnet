#! /bin/bash

. ./cmd.sh || exit 1 
. ./path.sh || exit 1 

mkdir -p data/{train,val,dev5_test}/split8


run.pl --gpu 1 JOB=1:8 exp/make_video_feats/dev5_test/log/vit/train.JOB.log \
    ./wrapper.sh JOB 
