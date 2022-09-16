import sys
import os
import string

ref_file = sys.argv[1]
hyp_file = sys.argv[2]
option = sys.argv[3] if len(sys.argv) > 3 else ""

## Option can be punc or lower

utt2spk_file = "dump/dev5_test/ref_utt2spk"
# id_file = "result_longaudio/ids"



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

        #.replace("<HOST>", "")
        #.replace("<GUEST>", "")

with open(hyp_file, "r") as f:
    hyp2text = {
        line.strip()
        .split(" ")[0]: " ".join(line.strip().split(" ")[1:])
        .replace("<sos/eos>", "")
        .strip()
        for line in f.readlines()
    }

        #.replace("<HOST>", "")
        #.replace("<GUEST>", "")
with open(utt2spk_file, "r") as f:
    utt2spk = {
        line.strip().split(" ")[0]: line.strip().split(" ")[1] for line in f.readlines()
    }

ids = [k for k,_ in hyp2text.items() if k in ref2text]
if option == "lower":
    final_hyp2text = {k: hyp2text[k].lower() for k in ids}
    final_ref2text = {k: ref2text[k].lower() for k in ids}
    tag = "l"
elif option == "punc":
    final_hyp2text = {
        k: hyp2text[k].translate(str.maketrans("", "", string.punctuation)).lower()
        for k in ids
    }
    final_ref2text = {
        k: ref2text[k].translate(str.maketrans("", "", string.punctuation)).lower()
        for k in ids
    }
    tag = "p"
else:
    final_hyp2text = {k: hyp2text[k] for k in ids}
    final_ref2text = {k: ref2text[k] for k in ids}
    tag = ""

with open(ref_file.replace("asr", tag + "asr") + ".trn", "w") as f:
    f.write(
        "\n".join(
            [
                "{} ({}-{})".format(final_ref2text[key], key, utt2spk[key])
                for key, _ in final_ref2text.items()
            ]
        )
    )

with open(hyp_file.replace("asr", tag + "asr") + ".trn", "w") as f:
    f.write(
        "\n".join(
            [
                "{} ({}-{})".format(final_hyp2text[key], key, utt2spk[key])
                for key, _ in final_hyp2text.items()
            ]
        )
    )
