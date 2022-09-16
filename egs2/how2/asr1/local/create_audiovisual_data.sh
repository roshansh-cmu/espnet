#! /bin/bash


set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=1

. ./db.sh
. ./path.sh
. ./cmd.sh


# url to download iwslt 2019 test set
base_url=/ocean/projects/iri120008p/roshansh/corpora/how2



[ -f ${base_url}/video.list ] || ls ${base_url}/all_videos/ | grep mp4 >  ${base_url}/video.list

mkdir -p data
mkdir -p data/{train,val,dev5_test}
for set in train val dev5_test; do 
    cp data/all/* data/${set}/
    awk -F '.' 'NR==FNR{a[$1]; next} $1 in a' ${base_url}/video.list ${base_url}/300h/${set}.list  > data/${set}/videos.list 
    cat data/${set}/videos.list | awk -F ' ' -v x=${base_url} '{$2=$1;print $1,x"/all_videos/"$2".mp4"}' > data/${set}/vad.scp
    cat data/${set}/vad.scp | awk -F ' ' '{print $1,"ffmpeg -i",$2,"-f wav -ar 16000 -ac 1 -vn - |"}' > data/${set}/wav.scp 
    utils/fix_data_dir.sh data/${set}
    mv data/${set}/vad.scp data/${set}/video.scp
done   
