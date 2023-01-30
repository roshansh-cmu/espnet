import os
import sys 
import numpy as np
from kaldiio import WriteHelper
import tqdm 

base_dir = "dump/extracted/"
dirs = [sys.argv[1]]
## "train_multilayer", "valid_multilayer", "test_multilayer"

for d in dirs:
    with open(os.path.join(base_dir, d, "feats_npy.scp"), "r") as f:
        m = {line.strip().split()[0]: line.strip().split()[1] for line in f.readlines()}
    with WriteHelper(
        "ark,scp:{0}.ark,{0}.scp".format(os.path.join(base_dir, d, "feats"))
    ) as writer:
        for k, v in tqdm.tqdm(m.items()):
            ar = np.load(os.path.join(v))
            permuted_combined = np.transpose(ar, (1, 0, 2)).reshape(ar.shape[1], -1)
            writer(k, permuted_combined)
