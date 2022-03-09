import kaldiio
from kaldiio import WriteHelper
import numpy as np
import os
import glob

feats_dir = "data/action_numpy"
vid2feats = {}
for file in os.listdir(feats_dir):
    if ".npy" in file:
        # data = np.load(os.path.join(feats_dir, file))
        fname = file.split(os.sep)[-1].split(".")[0]
        # print(f"{fname}")
        vid2feats[fname] = os.path.join(feats_dir, file)

# vid2feats = {
#     file.split(os.sep)[-1].split(".")[0]: np.load()
#     for file in os.listdir(feats_dir)
#     if ".npy" in file
# }

print(f"Read {len(list(vid2feats.keys()))} files ")
print(os.listdir("data"))
for data_dir in os.listdir("data"):
    print(data_dir)
    if not os.path.isdir(os.path.join("data", data_dir)):
        print("Is not dir {} no wav.scp".format(data_dir))
        continue
    if not os.path.isfile(os.path.join("data", data_dir, "wav.scp")):
        print("In Dir {} no wav.scp".format(data_dir))
        continue
    with open(os.path.join("data", data_dir, "wav.scp"), "r") as f:
        dataset_keys = [line.strip().split(" ")[0] for line in f.readlines()]
        common_keys = [k for k in dataset_keys if k in vid2feats]
        missing_keys = [k for k in dataset_keys if k not in common_keys]
        print(
            f"{data_dir} Missing Keys {len(missing_keys)} {len(common_keys)} {len(dataset_keys)} {missing_keys}"
        )
    if os.path.isfile(os.path.join("data", data_dir, "segments")):
        with open(os.path.join("data", data_dir, "segments"), "r") as f:
            utt_keys = [
                (line.strip().split(" ")[0], line.strip().split(" ")[1])
                for line in f.readlines()
                if line.strip().split(" ")[1] in common_keys
            ]
            utt2feats = {k[0]: vid2feats[k[1]] for k in utt_keys}
    else:
        utt2feats = {k: vid2feats[k] for k in common_keys}

    out_file = os.path.join("data", data_dir, "vad.scp")
    with open(out_file, "w") as f:
        f.write("\n".join(["{} {}".format(k, v) for k, v in utt2feats.items()]))
