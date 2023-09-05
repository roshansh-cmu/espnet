#! /bin/bash

data_feats=dump/fbank_pitch
dset=dev5_test_sumseg30
asr_task=asr_block
inference_bin_tag=
inference_tag=decode_sum_block_inference_v2
asr_exp=exp/asr_frompt_enchan_blockwise_block1k
start_nj=
end_nj=
batch_size=1
_ngpu=1
inference_config=conf/decode_sum_10s.yaml
python=python3
_scp=feats.scp
_type=kaldi_ark
inference_asr_model=valid.final_acc.best.pth


. utils/parse_options.sh

_data="${data_feats}/${dset}"
_dir="${asr_exp}/${inference_tag}/${dset}"
_logdir="${_dir}/logdir"
mkdir -p "${_logdir}"
_opts="--config ${inference_config} "

            run.pl JOB=${start_nj}:${end_nj} "${_logdir}"/asr_inference.JOB.log \
                ${python} -m espnet2.bin.${asr_task}_inference${inference_bin_tag} \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --asr_train_config "${asr_exp}"/config.yaml \
                    --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} || { cat $(grep -l -i error "${_logdir}"/asr_inference.*.log) ; exit 1; }
