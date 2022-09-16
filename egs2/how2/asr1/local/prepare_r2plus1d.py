"""

This file contains the visual feature extractor that can extract Resnet , VIT, R2Plus1D features
## R2Plus1D_18_Weights.KINETICS400_V1, resnext50_32x4d, ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
Resnext features are 2048 dimensional per frame
VIT features are 1024 dimensional per frame
R2Plus1D features are 512 dimensional per video 
"""

import logging
import os
import sys
import time

import numpy as np
import torch
import torchvision
from kaldiio import WriteHelper
from torchvision.io.video import read_video, read_video_timestamps
from torchvision.models import resnet, vision_transformer
from torchvision.models.video import R3D_18_Weights, r3d_18

# dsets = ["train", "val", "dev5_test"]
dsets = ["dev5_test"]

feature_sets = ["r2plus1d", "resnext32", "vit_l16"]

feature_sets = ["r2plus1d"]

BATCH_EVERY = 18
job_index = sys.argv[1]

print(f"JOB {job_index} | Launched task {job_index} ")


## Initialize Models, Weights and Pre-processors
feature2models = {
    # "vit_l16": torchvision.models.vit_l_16(
    #     weights=vision_transformer.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
    # ),
    # "resnext32": torchvision.models.resnext101_32x8d(
    #     weights=resnet.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
    # )
    # ,
    "r2plus1d": r3d_18(weights=R3D_18_Weights.DEFAULT),
}
print(f" JOB {job_index} |Initialized Models ")
feature2preprocess = {
    # "vit_l16": vision_transformer.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms(),
    # "resnext32": torchvision.models.resnet.ResNeXt101_32X8D_Weights.IMAGENET1K_V1.transforms(),
    "r2plus1d": R3D_18_Weights.DEFAULT.transforms(),
}
print(f"JOB {job_index} | Initialized Pre-processors ")

for k, v in feature2models.items():
    if k == "vit_l16":
        v.heads = torch.nn.Identity()
    elif k == "resnext32":
        v.fc = torch.nn.Identity()
    elif k == "r2plus1d":
        v.fc = torch.nn.Identity()
    if torch.cuda.is_available():
        v.cuda()
        feature2preprocess[k].cuda()
    v.eval()
print(f"JOB {job_index} | Removed penultimate classifier layers ")


for dset in dsets:
    video_scp = os.path.join("data", dset, "split8", f"video.{job_index}.scp")
    segments_file = os.path.join("data", dset, "segments")

    with open(segments_file, mode="r", encoding="utf-8") as f:
        vid2segments = {}
        nsegs = 0
        for line in f.readlines():
            nsegs += 1
            utt_id, vid_id, start, stop = line.strip().split(" ")
            if vid_id in vid2segments:
                vid2segments[vid_id].append([utt_id, float(start), float(stop)])
            else:
                vid2segments[vid_id] = [[utt_id, float(start), float(stop)]]
    print(f"JOB {job_index} | Read Segments file ")

    with open(video_scp, mode="r", encoding="utf-8") as f:
        print(f"JOB {job_index} | Read {video_scp}")
        out_path = os.path.join("dump", "fbank_pitch", dset, "video")
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        for feature_set in feature_sets:
            print(f"JOB {job_index} | Feature {feature_set}")

            feature_dump_directory = os.path.join(out_path, feature_set)
            if not os.path.isdir(feature_dump_directory):
                os.mkdir(feature_dump_directory)

            outf = f"feats.{job_index}."
            with WriteHelper(
                f"ark,scp:{os.path.join(feature_dump_directory,outf+ 'ark')},{out_path+'_'+feature_set+'_'+outf+'scp'}"
            ) as writer:
                i = 0
                nutt = 0
                lines = f.readlines()
                nlines = len(lines)
                for line in lines:
                    video_id, vid_path = line.strip().split(" ")
                    i += 1
                    vid, _, video_metadata = read_video(
                        vid_path,
                        pts_unit="sec",
                        output_format="TCHW",
                    )
                    st_time = time.time()
                    vid_timestamps, video_fps = read_video_timestamps(
                        vid_path, pts_unit="sec"
                    )
                    vid_timestamps = np.array(vid_timestamps)
                    # print(
                    #         f"JOB {job_index} | Read video {video_id} in {time.time()-st_time}s"
                    #     )
                    vid = vid[0 : vid.shape[0] : 3, ::]
                    vid_timestamps = vid_timestamps[0 : vid_timestamps.shape[0] : 3]
                    for (utt_id, start, stop) in vid2segments[video_id]:
                        st_time = time.time()
                        output = []
                        if len(vid_timestamps) > len(vid):
                            vid_timestamps = vid_timestamps[
                                : len(vid) - len(vid_timestamps)
                            ]
                        elif len(vid_timestamps) < len(vid):
                            vid_timestamps = (
                                vid_timestamps
                                + vid_timestamps[len(vid_timestamps) - len(vid) :]
                            )
                        assert len(vid) == len(
                            vid_timestamps
                        ), f"Vidlen {len(vid)} TS {len(vid_timestamps)}"
                        vid_input = vid[
                            np.all(
                                [vid_timestamps >= start, vid_timestamps <= stop],
                                axis=0,
                            )
                        ]
                        vid_input = vid_input[0 : vid_input.shape[0] : 3, ::]
                        # print(
                        #     f"JOB {job_index} | Segmented video {utt_id} in {time.time()-st_time}s"
                        # )
                        if vid_input.shape[0] > BATCH_EVERY:
                            for i in range(0, vid_input.shape[0], BATCH_EVERY):
                                vid_segment = (
                                    vid_input[i : i + BATCH_EVERY, ::]
                                    if i + BATCH_EVERY <= vid.shape[0]
                                    else vid_input[i:, ::]
                                )
                                if torch.cuda.is_available():
                                    vid_segment = vid_segment.cuda()
                                vid_segment = vid_segment.unsqueeze(0)
                                with torch.no_grad():
                                    vid_segment = feature2preprocess[feature_set](
                                        vid_segment
                                    )
                                    feats = feature2models[feature_set](vid_segment)
                                if feature_set == "r2plus1d":
                                    feats = feats.reshape(1, -1)
                                output.append(feats.cpu())
                            output = torch.cat(output, dim=0)
                        else:
                            with torch.no_grad():
                                if torch.cuda.is_available():
                                    vid_input = vid_input.cuda()
                                vid_input = vid_input.unsqueeze(0)
                                vid_input = feature2preprocess[feature_set](vid_input)
                                # print(
                                #     f"JOB {job_index} | Pre-processed video {utt_id} in {time.time()-st_time}s"
                                # )
                                output = feature2models[feature_set](vid_input)
                        writer(utt_id, output.cpu().detach().numpy())
                        # print(
                        #     f"JOB {job_index} | Finished Feature Extraction {utt_id} in {time.time()-st_time}s"
                        # )

                        nutt += 1
                        if nutt % 1000 == 0:
                            print(f"JOB {job_index} | Finished {nutt} segments")
                    if i % 500 == 0:
                        print(
                            f"JOB {job_index} | Finished {i}/{nlines} videos | {nutt} videos"
                        )
