#! /bin/bash



exp_dir=$1 
echo "EXP DIR is $exp_dir"

for x in $(ls ${exp_dir}/ | grep decode ); do
    decode_dir=${exp_dir}/${x}
    for y in $(ls ${decode_dir}/ | grep test); do 
        dir=${decode_dir}/${y}
        score_dir=${dir}/score_wer
        new_score_dir=$dir/score_fwer
        mkdir -p $new_score_dir

        if [[ $y == "dev5_test" ]]; then 
            segments_file=data/dev5_test/segments
            ref_text=data/dev5_test/text
        elif [[ $y == "dev5_test_seg20_nolap" ]]; then 
            segments_file=data/dev5_test_seg20_nolap/segments
            ref_text=data/dev5_test_seg20_nolap/text
        elif  [[ $y == "dev5_test_seg15_nolap" ]]; then
            segments_file=data/dev5_test_seg15_nolap/segments
            ref_text=data/dev5_test_seg15_nolap/text
        fi 

        python local/replace_contractions.py ${score_dir}/hyp.trn > ${new_score_dir}/hyp.trn 
        python local/replace_contractions.py ${score_dir}/ref.trn  > ${new_score_dir}/ref.trn 
        python local/replace_contractions.py ${dir}/text  > ${dir}/hyp_filtered 
        python local/replace_contractions.py ${ref_text}  > ${dir}/ref_filtered 

        echo "Finished data filtering for ${dir}; Text file is ${ref_text}; Segments file is ${segments_file}"

        sclite -r "$new_score_dir/ref.trn" trn -h "$new_score_dir/hyp.trn" trn -i rm -o all stdout > ${new_score_dir}/result.txt

        ## Video level Scoring 
        conv_score_dir=$dir/score_ver
        mkdir -p $conv_score_dir

        python local/create_trn_files_conv.py ${dir}/ref_filtered ${dir}/hyp_filtered ${segments_file} $conv_score_dir/ref.trn $conv_score_dir/hyp.trn       
        sclite -r "$conv_score_dir/ref.trn" trn -h "$conv_score_dir/hyp.trn" trn -i rm -o all stdout > ${conv_score_dir}/result.txt
        done 
done 


    
    







    







