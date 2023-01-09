#! /bin/bash

# Copyright 2022 Roshan Sharma (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script prepares the trim feature directories based on a specified trim length 
"""
import os
import sys
import shutil 
from kaldiio import ReadHelper, WriteHelper

data_dir = sys.argv[1]
trim_length = int(sys.argv[2]) if len(sys.argv) > 2 else 100

out_dir = data_dir + f"seg{trim_length}"

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
shutil.copytree(data_dir, out_dir)

with ReadHelper(f"scp:{data_dir}/feats.scp") as reader, WriteHelper(
    f"ark,scp:{out_dir}/feats.ark,{out_dir}/feats.scp"
) as writer:
    for key, mat in reader:
        writer(key, mat[:(trim_length*100), :])





