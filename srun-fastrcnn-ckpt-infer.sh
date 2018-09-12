#!/usr/bin/env sh
echo "This file requires a ckpt to load from, please train a rcnn ckpt first" 
exit
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=Test --mpi=pmi2 --gres=gpu:1 --job-name=rcnn_test_ckpt --kill-on-bad-exit=1 python tools/infer_simple.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml --load_ckpt Outputs/e2e_faster_rcnn_R-50-FPN_1x/current_9w_step/ckpt/model_step19999.pth --image_dir test  --output_dir prediction 2>&1|tee log/rcnn/test-ckpt-infer-$now.log &
