#! /bin/bash


## sed "s=\#=hashtag =g"
## root=dump/extracted
root=data
newsub=_filt
dirs="train valid test"

for dir in $dirs; do
    echo "Filtering $dir"
    [ -z ${newsub} ] || rm -rf ${root}/${dir}${newsub}
    [ -f ${root}/${dir}${newsub}/text ] || cp -r ${root}/${dir} ${root}/${dir}${newsub}
    cut -d ' ' -f 1 ${root}/${dir}${newsub}/text > ${root}/${dir}${newsub}/text.ids
    
    cut -d ' ' -f 2- ${root}/${dir}${newsub}/text | sed 's=&quot;==g' | sed "s=&#39;='=g" | sed "s=&#x27;='=g" |\
     sed 's/([^()]*)//g' | sed "s=\[SEP\]=\&=g" | sed -e 's/\[[^][]*\]//g' | sed "s=&amp;= and =g" |\
     sed "s=\&lt;\/i\&gt;==g" | sed "s=\&lt;i\&gt;==g" | sed "s=\&gt;=is greater than=g" | sed "s=|==g" |\
     sed "s=Youre =You are =g" | sed "s= youre = you are =g" | sed "s=Its =It is =g" | sed "s= its = it is =g" |\
     sed "s=Youve =You have =g" | sed "s= youve = you have =g" | sed "s=Im =I am =g" | sed "s= im = i am =g" |\
     sed "s=Theyve =They have =g" | sed "s= theyve = they have =g" | sed "s=Thats =That is =g" | sed "s=thats =that is =g" |\
     sed "s=Theyre =They are =g" | sed "s= theyre = they are =g" | sed "s=Hes =He is =g" | sed "s=hes =he is =g" |\
     sed "s= Its = It is =g" | sed "s= its = it is =g" | sed "s=Theres =There is =g" | sed "s=theres =there is =g" |\
     sed "s= dont = do not =g" | sed "s= Dont = Do not =g" | sed "s= lets = let us =g" | sed "s=Lets =Let us =g" |\
     sed "s=\.\.\.==g" | sed 's=\. \.=\.=g' | python local/normalize_whisper_text.py > ${root}/${dir}${newsub}/text_filt

    paste -d ' ' ${root}/${dir}${newsub}/text.ids ${root}/${dir}${newsub}/text_filt | LC_ALL=C sort > ${root}/${dir}${newsub}/text

    cut -d '&' -f1 ${root}/${dir}${newsub}/text_filt > ${root}/${dir}${newsub}/title${newsub} 
    cut -d '&' -f2 ${root}/${dir}${newsub}/text_filt | sed "s=^ ==g" > ${root}/${dir}${newsub}/abstract${newsub}

    paste -d ' ' ${root}/${dir}${newsub}/text.ids ${root}/${dir}${newsub}/title${newsub} > ${root}/${dir}${newsub}/title
    paste -d ' ' ${root}/${dir}${newsub}/text.ids ${root}/${dir}${newsub}/abstract${newsub} > ${root}/${dir}${newsub}/abstract 

    utils/validate_data_dir.sh --no-feats ${root}/${dir}${newsub}
done
