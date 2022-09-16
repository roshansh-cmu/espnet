import os
import string
import sys

ref_file = sys.argv[1]
hyp_file = sys.argv[2]
segments_file = sys.argv[3]
out_ref_file = sys.argv[4]
out_hyp_file = sys.argv[5]

# os.path.join(os.path.dirname(os.path.dirname(ref_file)), "segments")
utt2spk_file = os.path.join(os.path.dirname(segments_file), "utt2spk")
# id_file = os.path.join(os.path.dirname(os.path.dirname(ref_file)), "ids")

## Option can be punc or lower

with open(segments_file, "r") as f:
    dia2segs = {}
    for line in f.readlines():
        uttid, dia, start, end = line.strip().split(" ")
        if dia in dia2segs:
            dia2segs[dia].append([uttid, float(start), float(end)])
        else:
            dia2segs[dia] = [[uttid, float(start), float(end)]]

for dia, segs in dia2segs.items():
    segs.sort(key=lambda x: x[1])

# with open(id_file, "r") as f:
#     ids = [x.strip() for x in f.readlines()]


with open(ref_file, "r") as f:
    ref2text = {
        line.strip()
        .split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        .replace("<sos/eos>", "")
        .strip()
        for line in f.readlines()
    }

with open(hyp_file, "r") as f:
    hyp2text = {
        line.strip()
        .split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        .replace("<sos/eos>", "")
        .strip()
        for line in f.readlines()
    }

with open(utt2spk_file, "r") as f:
    utt2spk = {
        line.strip().split(" ")[0]: line.strip().split(" ")[1] for line in f.readlines()
    }
ids = [k for k, _ in hyp2text.items() if k in ref2text]

final_hyp2text = {k: hyp2text[k] for k in ids}
final_ref2text = {k: ref2text[k] for k in ids}
tag = ""


hyp_list = []
ref_list = []
pun_ref_list = []
pun_hyp_list = []
sa_ref_list = []
sa_hyp_list = []
sa_hyp_list = []
sa_ref_list = []
for dia, segs in dia2segs.items():
    hyp_conv = " ".join([final_hyp2text[x[0]] for x in segs])
    ref_conv = " ".join([final_ref2text[x[0]] for x in segs])
    segs = ref_conv.split(" <")
    hyp_list.append("{} ({}-{})".format(hyp_conv, dia, dia))
    ref_list.append("{} ({}-{})".format(ref_conv, dia, dia))

with open(out_ref_file, "w") as f:
    f.write("\n".join(ref_list))

with open(out_hyp_file, "w") as f:
    f.write("\n".join(hyp_list))

# with open(ref_file.replace("asr", tag + "pasr") + ".trn", "w") as f:
#     f.write("\n".join(pun_ref_list))

# with open(hyp_file.replace("asr", tag + "pasr") + ".trn", "w") as f:
#     f.write("\n".join(pun_hyp_list))

# with open(ref_file.replace("asr", tag + "sasr") + ".trn", "w") as f:
#     f.write("\n".join(sa_ref_list))

# with open(hyp_file.replace("asr", tag + "sasr") + ".trn", "w") as f:
#     f.write("\n".join(sa_hyp_list))
