#!/usr/bin/env bash

set -e
set -u
set -o pipefail

train_set="train_seg15"
valid_set="val_seg15"
test_sets="dev5_test"

asr_config=conf/train_asr_cf_fbank.yaml
inference_config=conf/decode.yaml

feats_type=fbank_pitch

token_type=bpe

nlsyms=data/nlsyms

nbpe=1000
bpe_nlsyms="[hes]"

use_lm=false

./asr.sh                                        \
    --nj 32 \
    --ngpu 8 \
    --stage 10 \
    --stop_stage 11 \
    --asr_stats_dir exp/asr_stats_fbank_seg15 \
    --pretrained_model "exp/asr_conformer_base/valid.acc.best.pth" \
    --ignore_init_mismatch true \
    --asr_args "--use_wandb true --wandb_project how2_asilomar --wandb_name conformer_seg15" \
    --feats_type ${feats_type}                  \
    --token_type ${token_type}                  \
    --nbpe ${nbpe}                              \
    --nlsyms_txt ${nlsyms}                      \
    --bpe_nlsyms ${bpe_nlsyms}                  \
    --use_lm ${use_lm}                          \
    --asr_config "${asr_config}"                \
    --inference_config "${inference_config}"          \
    --train_set "${train_set}"                  \
    --valid_set "${valid_set}"                  \
    --test_sets "${test_sets}"                  \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
