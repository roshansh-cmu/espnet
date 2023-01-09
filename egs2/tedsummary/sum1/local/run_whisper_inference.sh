#! /bin/bash

. ./cmd.sh 
. ./path.sh 

stage=1
cmd=run.pl 
njobs_per_gpu=15
ngpus=1
dumpsets="valid_vad test_vad train_vad"

. parse_options.sh || exit 1;

nj=$((njobs_per_gpu * ngpus))
echo "Running with ${ngpus} GPUs and ${njobs_per_gpu} jobs per GPU = ${nj} jobs in parallel "


for dset in ${dumpsets}; do 
    _data=data/${dset}
    logdir=data/${dset}/log
    mkdir -p $logdir
    
    key_file=${_data}/wav.scp
    split_scps=""
    for n in $(seq "${nj}"); do
        split_scps+=" ${logdir}/keys.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}
    
    mkdir -p ${logdir}/logfiles  

    ${cmd} JOB=1:$njobs_per_gpu "${logdir}/logfiles/dump_feats.JOB.log" \
        python local/perform_whisper_inference.py ${logdir}/keys.JOB.scp ${logdir}/text.JOB

    for n in $(seq "${nj}"); do
        cat ${logdir}/text.${n}
    done > ${_data}/text_whisper
done