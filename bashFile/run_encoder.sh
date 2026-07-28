#!/bin/sh
# Requires: pip install . (or pip install stampone) from the repo root,
# which installs the stampone-encode console script.
export MASTER_PORT=6036
echo MASTER_PORT=${MASTER_PORT}

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=$1
MODE=$2 # default or None
GPUS=$3

if [ -z "$1" ]; then
   GPUS=1
fi

echo $DATASET $MODE $GPUS

if [[ $MODE == default ]]; then
	echo "==> default Setting"
    stampone-encode
else
    echo "==> Setup Setting"
    stampone-encode --gpu_devices 0 \
                       --detector FaceDetection \
                       --original_images 'path_to_dataset/' \
                       --save_dir "./results/" \
                       --random_message False \
                       --message "Visteam" \
                       --BCH_BITS 25 \
                       --BCH_POLYNOMIAL 487 \
                       --secret_size 256