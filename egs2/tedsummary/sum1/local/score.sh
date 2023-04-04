#!/usr/bin/env bash
# Copyright 2021 Carnegie Mellon University (Author : Roshan Sharma)

## begin configuration section.
data=data/test_filt
tag="_newtext"

## end configuration section.


[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir>"
  exit 1;
fi


asr_expdir=$1
for decode_dir in $(ls ${asr_expdir}/  | grep decode ); do
	for name in $(ls -d ${asr_expdir}/${decode_dir}/* | grep test ); do
        
		dir=${name}
	    echo "Scoring ${name}"
		python pyscripts/utils/score_summarization.py $data/text $dir/text $(echo $dir | sed 's/exp//g') > $dir/result${tag}.sum

		# ## Filtering text and scoring
		for text_file in $data/text $dir/text; do
			echo ${text_file}
			cut -d ' ' -f1 ${text_file} > "${text_file}".ids
			cut -d ' ' -f 2- ${text_file} | sed 's=&quot;==g' | sed "s=&#39;='=g" | sed "s=&#x27;='=g" |\
				sed 's/([^()]*)//g' | sed "s=\[SEP\]=\&=g" | sed -e 's/\[[^][]*\]//g' | sed "s=&amp;= and =g" |\
				sed "s=\&lt;\/i\&gt;==g" | sed "s=\&lt;i\&gt;==g" | sed "s=\&gt;=is greater than=g" | sed "s=|==g" |\
				sed "s=Youre =You are =g" | sed "s= youre = you are =g" | sed "s=Its =It is =g" | sed "s= its = it is =g" |\
				sed "s=Youve =You have =g" | sed "s= youve = you have =g" | sed "s=Im =I am =g" | sed "s= im = i am =g" |\
				sed "s=Theyve =They have =g" | sed "s= theyve = they have =g" | sed "s=Thats =That is =g" | sed "s=thats =that is =g" |\
				sed "s=Theyre =They are =g" | sed "s= theyre = they are =g" | sed "s=Hes =He is =g" | sed "s=hes =he is =g" |\
				sed "s= Its = It is =g" | sed "s= its = it is =g" | sed "s=Theres =There is =g" | sed "s=theres =there is =g" |\
				sed "s= dont = do not =g" | sed "s= Dont = Do not =g" | sed "s= lets = let us =g" | sed "s=Lets =Let us =g" |\
				sed "s=\.\.\.==g" | sed 's=\. \.=\.=g' | python local/normalize_whisper_text.py  > ${text_file}.text_filt
			paste -d ' ' ${text_file}.ids ${text_file}.text_filt > ${text_file}.filt
			cat ${text_file}.text_filt |tr '[:upper:]' '[:lower:]' | tr -d '[:punct:]' > ${text_file}.filt.nopunc
			paste -d ' ' ${text_file}.ids ${text_file}.filt.nopunc > ${text_file}.filtnopunc
		done
		## Score Filtered
		python pyscripts/utils/score_summarization.py $data/text.filt $dir/text.filt $(echo $dir | sed 's/exp//g') > $dir/result_filt${tag}.sum
		python pyscripts/utils/score_summarization.py $data/text.filtnopunc $dir/text.filtnopunc $(echo $dir | sed 's/exp//g') > $dir/result_filt_nopunc${tag}.sum

	done
done
