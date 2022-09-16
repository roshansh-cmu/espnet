#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train_subset_50"
valid_set="val_av_res"
test_sets="dev5_test"

asr_config=conf/train_asr_av_res_tf_fbank_hierattn.yaml
inference_config=conf/decode.yaml

feats_type=fbank_pitch
video_feats_type=resnext32
token_type=bpe

nlsyms=data/nlsyms

nbpe=1000
bpe_nlsyms="[hes]"

use_lm=false

./asr_av.sh                                        \
    --nj 32 \
    --ngpu 8 \
    --stage 10 \
    --stop_stage 11 \
    --feats_type ${feats_type}                  \
    --token_type ${token_type}                  \
    --nbpe ${nbpe}                              \
    --nlsyms_txt ${nlsyms}                      \
    --bpe_nlsyms ${bpe_nlsyms}                  \
    --use_lm ${use_lm}                          \
    --asr_config "${asr_config}"                \
    --pretrained_model "../asr1/exp/asr_asr_base_subset/valid.acc.best.pth" \
    --ignore_init_mismatch true \
    --asr_args "--use_wandb true --wandb_project how2_asilomar --wandb_name av_resnext_subset_hierattn_novat" \
    --asr_stats_dir exp/asr_av_stats_subset_${video_feats_type} \
    --video_feats_type ${video_feats_type}      \
    --inference_config "${inference_config}"          \
    --train_set "${train_set}"                  \
    --valid_set "${valid_set}"                  \
    --test_sets "${test_sets}"                  \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
