#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh


dset=
gpu=
njobs_per_gpu=
mode=
dump_dir=
asr_model_dir=
cmd=run.pl 
logdir=
feats_type=multilayer 

. parse_options.sh || exit 1;

gpu_id=$((gpu-1))
sta=$(( ${gpu_id}*${njobs_per_gpu} +1 ))
end=$(( ${gpu_id}*${njobs_per_gpu} + ${njobs_per_gpu} ))

echo $gpu_id $sta $end

_data=dump/extracted/${dset}
CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd} JOB=${sta}:${end} "${logdir}/logfiles/${gpu_id}/dump_hubert.JOB.log" \
	python3 -m espnet2.bin.asr_encoder_dump \
	--feats_type ${feats_type} \
        --batch_size 1 \
        --ngpu 1 \
	--key_file ${logdir}/keys.JOB.scp \
        --data_path_and_name_and_type "${_data}/wav.scp,speech,sound" \
        --asr_train_config ${asr_model_dir}/config.yaml \
        --mode ${mode} \
        --dump_dir ${dump_dir} \
        --output_dir "${logdir}"  
