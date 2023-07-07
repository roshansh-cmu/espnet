#!/bin/bash

set -e
set -u
set -o pipefail

train_set="tr_2000h_sum"
valid_set="cv05_sum"
test_sets="dev5_test_sum"
asr_config=conf/train_sum_fnet.yaml
inference_config=conf/decode_sum_30s.yaml
inference_asr_model=valid.acc.best.pth 

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
    --asr_tag baseline_fnet \
    --feats_type ${feats_type} \
    --token_type ${token_type} \
    --nbpe ${nbpe} \
    --nlsyms_txt ${nlsyms} \
    --inference_nj 10 \
    --gpu_inference true \
    --ignore_init_mismatch true \
    --stage 11 \
    --stop_stage 13 \
    --asr_stats_dir exp/asr_stats_fbankpitch43 \
    --pretrained_model "exp/asr_train_asr_fnet_base_layernorm_extracted_en_bpe1000/valid.acc.best.pth:::ctc" \
    --inference_asr_model ${inference_asr_model} \
    --bpe_nlsyms ${bpe_nlsyms} \
    --use_lm ${use_lm} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --max_wav_duration "$mdur" \
    --bpe_train_text "data/${train_set}/text" "$@"
