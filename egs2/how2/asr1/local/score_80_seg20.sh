#! /bin/bash

. ./cmd.sh || exit 1
. ./path.sh || exit 1

ref_file=data/dev5_test/text
dir=results_seg20

for hyp_file in $( ls ${dir}/*_hyp ); do  
    #score_tag=$( basename $hyp_file | sed 's=_hyp==g') 
    #ref_file=${dir}/ref
    score_dir=${dir}/scoring_${score_tag}
    mkdir -p $score_dir
    
    ## Test1 : ASR WER
    cp $hyp_file $score_dir/asr_hyp
    cp $ref_file $score_dir/asr_ref
    
    python local/create_trn_files.py $score_dir/asr_ref $score_dir/asr_hyp 
    sclite -r "$score_dir/asr_ref.trn" trn -h "$score_dir/asr_hyp.trn" trn -i rm -o all stdout > ${score_dir}/asr_result.txt

    ## Test 2 Conv WER
    python local/create_trn_files_conv.py $score_dir/asr_ref $score_dir/asr_hyp
    sclite -r "$score_dir/casr_ref.trn" trn -h "$score_dir/casr_hyp.trn" trn -i rm -o all stdout > ${score_dir}/casr_result.txt
done
