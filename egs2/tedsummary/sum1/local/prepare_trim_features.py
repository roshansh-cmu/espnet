#! /bin/bash

# Copyright 2022 Roshan Sharma (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script prepares the trim feature directories based on a specified trim length 
"""
import os
import sys
import shutil 
import tqdm
from kaldiio import ReadHelper, WriteHelper

data_dir = sys.argv[1]
trim_length = int(sys.argv[2]) if len(sys.argv) > 2 else 100
index= sys.argv[3] if len(sys.argv) > 3 else ""
out_dir = data_dir + f"seg{trim_length}"

# if os.path.exists(out_dir):
#     shutil.rmtree(out_dir)
# shutil.copytree(data_dir, out_dir)

with ReadHelper(f"scp:{data_dir}/feats{index}.scp") as reader, WriteHelper(
    f"ark,scp:{out_dir}/feats{index}.ark,{out_dir}/feats{index}.scp"
) as writer:
    for key, mat in tqdm.tqdm(reader):
        writer(key, mat[:(trim_length*100), :])





