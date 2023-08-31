#!/bin/bash

set -e
set -u
set -o pipefail

train_set="tr_2000h_sumseg50"
valid_set="cv05_sumseg50"
test_sets="dev5_test_sumseg50"
asr_config=conf/train_sum_bass.yaml
inference_config=conf/decode_sum.yaml

feats_type=fbank_pitch

token_type=bpe

nlsyms=data/nlsyms
nbpe=1000
bpe_nlsyms="[hes]"

use_lm=false
mdur=100

## Run local/run_asr.sh to pretrain an ASR Model on How2, and fine-tune that model on Speech Summarization

./asr.sh \
    --lang en \
    --feats_type ${feats_type} \
    --token_type ${token_type} \
    --nbpe ${nbpe} \
    --nlsyms_txt ${nlsyms} \
    --stage 11 \
    --stop_stage 13 \
    --asr_stats_dir exp/asr_stats_fbank30_pitch_f43 \
    --pretrained_model exp/asr_baseline_10s/valid.acc.best.pth \
    --asr_args "--use_wandb true --wandb_project how2_blockwise --wandb_name baseline_30s_encatt_block10" \
    --bpe_nlsyms ${bpe_nlsyms} \
    --use_lm ${use_lm} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --max_wav_duration "$mdur" \
    --bpe_train_text "data/${train_set}/text" "$@"
