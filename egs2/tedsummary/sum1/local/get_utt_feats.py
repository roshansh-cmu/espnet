import sys
import os
from kaldiio import ReadHelper, WriteHelper
import tqdm


long_feats_scp = sys.argv[1]
segments_file = sys.argv[2]
long_rec_lengths = sys.argv[3]
outfile = sys.argv[4]


rec2segments = {}
i = 0
with open(segments_file, "r") as f:
    for line in f.readlines():
        uid, recid, start, end = line.strip().split(" ")
        start = float(start)
        end = float(end)
        if recid in rec2segments:
            rec2segments[recid].append([uid, start, end])
        else:
            rec2segments[recid] = [[uid, start, end]]
        i +=1 
print(f"Loaded {i} segments from {segments_file}.")


with open(long_rec_lengths, "r") as f:
    rec2len = {
        line.strip().split(" ")[0]: float(line.strip().split(" ")[1])
        for line in f.readlines()
    }


with ReadHelper("scp:" + long_feats_scp) as reader, WriteHelper(
    f"ark,scp:{outfile}.ark,{outfile}.scp"
) as writer:
    for key, mat in tqdm.tqdm(reader):
        if key in rec2segments and key in rec2len:
            for uid, start, end in rec2segments[key]:
                start_frame = int(float(start / rec2len[recid]) * mat.shape[0])
                end_frame = int(float(end / rec2len[recid]) * mat.shape[0])
                writer(uid, mat[start_frame:end_frame, :])
        #else:
        #    print(f"Missing utt_ids {key}")
