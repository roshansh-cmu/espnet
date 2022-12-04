#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys

import pandas as pd
import json
import random

random.seed(5)

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [tedsum_root]")
    sys.exit(1)
tedsum_root = sys.argv[1]
data = json.load(open(tedsum_root+"/ted_summary_correct_transcript.json","rb"))

summ_dict={}
wav_scp_dict={}
speaker_dict={}
transcript_dict={}
for k in data:
    if "abst" not in data[k]:
        print(k)
        continue
    summ_dict[k]=data[k]["title"].strip().encode("ascii", "ignore").decode().replace("\n"," ")+" [SEP] "+data[k]["abst"].strip().encode("ascii", "ignore").decode().replace("\n"," ")
    summ_dict[k]=summ_dict[k].replace("\r"," ")
    if " and " in data[k]["speaker"]:
        speaker_dict[k]="Two_Speaker_"+data[k]["speaker"].strip().replace(" ","-")
    elif " + " in data[k]["speaker"]:
        speaker_dict[k]="Two_Speaker_"+data[k]["speaker"].strip().replace(" ","-")
    elif " with " in data[k]["speaker"]:
        speaker_dict[k]="Two_Speaker_"+data[k]["speaker"].strip().replace(" ","-")
    elif ", " in data[k]["speaker"]:
        speaker_dict[k]="Two_Speaker_"+data[k]["speaker"].strip().replace(" ","-")
    else:
        speaker_dict[k]="Speaker_"+data[k]["speaker"].strip().replace(" ","-")
    if "key" not in data[k]:
        print(k)
        exit()
    wav_scp_dict[k]=tedsum_root+"/audio_data2/"+data[k]["key"]+".mp4"
    transcript_dict[k]=data[k]["transcript"].strip().encode("ascii", "ignore").decode().replace("\n"," ").replace("\r"," ")

key_arr=list(data.keys())
random.shuffle(key_arr)
train_key_dict={k:1 for k in key_arr[:int(0.8*len(key_arr))]}
valid_key_dict={k:1 for k in key_arr[int(0.8*len(key_arr)):int(0.9*len(key_arr))]}
test_key_dict={k:1 for k in key_arr[int(0.9*len(key_arr)):]}

dir_dict = {
    "train": train_key_dict,
    "valid": valid_key_dict,
    "test": test_key_dict,
}

for x in dir_dict:
    with open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join("data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join("data", x, "transcript"), "w"
    ) as transcript_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f:

        text_f.truncate()
        transcript_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for k in summ_dict:
            if k in dir_dict[x]:
                utt_id=str(speaker_dict[k])+"_"+k
                text_f.write(utt_id + " " + summ_dict[k] + "\n")
                wav_scp_f.write(utt_id + " " + wav_scp_dict[k] + "\n")
                utt2spk_f.write(utt_id + " " + str(speaker_dict[k]) + "\n")
                transcript_f.write(utt_id + " " + transcript_dict[k] + "\n")
