import os
import sys

import numpy as np
from datasets import load_metric
from nlgeval import NLGEval, compute_metrics
import spacy
from spacy import displacy
import editdistance

ref_file = sys.argv[1]
hyp_file = sys.argv[2]

NER = spacy.load("en_core_web_sm")

with open(ref_file, "r") as f:
    ref_dict = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:]).split(" [SEP] ")[0]
        for line in f.readlines()
    }

with open(hyp_file, "r") as f:
    hyp_dict = {
        line.strip().split(" ")[0]: " ".join(line.strip().split(" ")[1:]).split(" [SEP] ")[0]
        for line in f.readlines()
    }

keys = [k for k, v in hyp_dict.items()]
labels = [ref_dict[k] for k, _ in hyp_dict.items()]
decoded_preds = [v for k, v in hyp_dict.items()]
overlap_dict={}
count_dict={}
for (key, ref, hyp) in zip(keys, labels, decoded_preds):
    text1= NER(ref)
    for word in text1:
        if word.pos_ not in overlap_dict:
            overlap_dict[word.pos_]=0
            count_dict[word.pos_]=0
        if word.text in hyp:
            overlap_dict[word.pos_]+=1
        else:
            for k in hyp.split():
                if editdistance.eval(k, word.text)/len(word.text)<0.6:
                    overlap_dict[word.pos_]+=1
        count_dict[word.pos_]+=1
    # break

print(count_dict)
print(overlap_dict)
for k in overlap_dict:
    overlap_dict[k]/=count_dict[k]
print(overlap_dict)
