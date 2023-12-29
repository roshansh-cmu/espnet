#!/usr/bin/env bash
# Copyright 2021 Carnegie Mellon University (Author : Roshan Sharma)

## begin configuration section.
ref_file=data/dev5_test_sum/text
inference_tag=decode
# end configuration section.


[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir>"
  exit 1;
fi


asr_expdir=$1

for decode_dir in $(ls -d ${asr_expdir}/*/ | grep ${inference_tag}); do
	for test_dir in $(ls -d ${decode_dir}/*/); do
		for hyp_file in $(ls ${test_dir}/ | grep text); do
			hyp_file=${test_dir}/${hyp_file}
			pref=${hyp_file##*/}
			pref=$(echo ${pref} | sed 's=text==g')
    			python pyscripts/utils/score_summarization.py ${ref_file} ${hyp_file} $(echo $dir | sed 's/exp//g') > ${test_dir}/${pref}result.sum
	
		done
	done
done
