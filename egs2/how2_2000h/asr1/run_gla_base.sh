#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="tr_2000h_utt"
valid_set="cv05_utt"
test_sets="dev5_test_utt"

asr_config=conf/train_asr_conformer_gla.yaml
inference_config=conf/decode_asr.yaml

feats_type=fbank_pitch

token_type=bpe

nlsyms=data/nlsyms
nbpe=1000
bpe_nlsyms="[hes]"

use_lm=false


./asr.sh \
    --lang en \
    --feats_type ${feats_type} \
    --token_type ${token_type} \
    --nbpe ${nbpe} \
    --nlsyms_txt ${nlsyms} \
    --bpe_nlsyms ${bpe_nlsyms} \
    --use_lm ${use_lm} \
    --asr_config "${asr_config}" \
    --stage 11 \
    --asr_stats_dir exp/asr_stats_fbank43_utt \
    --asr_tag asr_fbank_fnet \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --max_wav_duration 30 \
    --bpe_train_text "data/${train_set}/text" "$@"
