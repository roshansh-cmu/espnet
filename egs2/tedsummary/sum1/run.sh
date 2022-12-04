#!/bin/bash

set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test"
asr_config=conf/train_sum_conformer_lf.yaml
inference_config=conf/decode_sum.yaml

feats_type="raw"

token_type=bpe

nlsyms=data/nlsyms
nbpe=500
bpe_nlsyms="[SEP]"

use_lm=false

## Run local/run_asr.sh to pretrain an ASR Model on How2, and fine-tune that model on Speech Summarization

./asr.sh \
    --lang en \
    --stage 2 \
    --stop_stage 4 \
    --feats_type ${feats_type} \
    --token_type ${token_type} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --nbpe ${nbpe} \
    --nlsyms_txt ${nlsyms} \
    --bpe_nlsyms ${bpe_nlsyms} \
    --use_lm ${use_lm} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --max_wav_duration 100000000000 \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" "$@"
