#! /bin/bash

. ./cmd.sh || exit 1 
. ./path.sh || exit 1 

mkdir -p data/{train,val,dev5_test}/split100


#run.pl --gpu 1 JOB=2:2 exp/make_video_feats/train/log/r2plus1d/video.JOB.log ./wrapper_vid.sh JOB 
#run.pl --gpu 1 JOB=1:8 exp/make_video_feats/train/log/r2plus1d/video.JOB.log ./wrapper_vid.sh JOB 
#run.pl --gpu 1 JOB=24:31 exp/make_video_feats/train/log/r2plus1d/video.JOB.log ./wrapper_vid.sh JOB 
run.pl --gpu 1 JOB=1:8 exp/make_video_feats/dev5_test/log/r2plus1d/video.JOB.log ./wrapper_vid.sh JOB 
