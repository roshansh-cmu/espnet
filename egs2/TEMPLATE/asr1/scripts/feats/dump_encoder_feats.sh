#! /bin/bash

. ./cmd.sh 
. ./path.sh 

stage=1
cmd=run.pl 
njobs_per_gpu=4
ngpus=1
dumpsets=
asr_model_dir=
dump_dir=
mode=frontend
feats_dim=1024 
feats_type=last

. parse_options.sh || exit 1;

nj=$((njobs_per_gpu * ngpus))
echo "Running with ${ngpus} GPUs and ${njobs_per_gpu} jobs per GPU = ${nj} jobs in parallel "
mkdir -p ${dump_dir}

if [ $stage -le 1 ]; then  
    echo "Stage 1: create extracted directory"
    mkdir -p dump/extracted
    rsync -ax --exclude=data dump/raw/ dump/extracted/
    for dset in $(ls -d dump/extracted/*/ | grep -v org ); do
        echo "extracted" > ${dset}/feats_type
        echo "${feats_dim}" > ${dset}/feats_dim
    done 
fi 

if [ $stage -le 2 ]; then  
    echo "Stage 2: Split data directories"
    for dset in ${dumpsets}; do 
        _data=dump/extracted/${dset}
        logdir=dump/extracted/${dset}/log
        mkdir -p $logdir
        
        key_file=${_data}/wav.scp
        split_scps=""
        for n in $(seq "${nj}"); do
            split_scps+=" ${logdir}/keys.${n}.scp"
        done
        utils/split_scp.pl "${key_file}" ${split_scps}
    done
fi 


if [ $stage -le 3 ]; then  
    echo "Stage 3: Extract features"
    for dset in ${dumpsets}; do 
        logdir=dump/extracted/${dset}/log
        mkdir -p ${logdir}/logfiles      
        mkdir -p ${dump_dir}/${dset}  
        ${cmd} JOB=1:$ngpus "${logdir}/logfiles/dump_feats.JOB.log" \
            ./scripts/feats/feats_dump_wrapper.sh \
            --dset ${dset} \
            --gpu JOB \
            --njobs_per_gpu ${njobs_per_gpu} \
            --mode ${mode} \
            --dump_dir ${dump_dir}/${dset} \
            --asr_model_dir ${asr_model_dir} \
            --logdir ${logdir} \
	    --feats_type ${feats_type} \
            --cmd ${cmd}
    done 
        cat ${logdir}/feats.*.scp | LC_ALL=C sort > dump/extracted/${dset}/feats.scp

fi
