#!/usr/bin/env sh
 
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=Test --mpi=pmi2 --gres=gpu:2 --job-name=rcnn_train --kill-on-bad-exit=1 python tools/train_net_step.py --nw 1 --dataset triangle_2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml   --use_tfboard --bs 2 --iter_size 1 2>&1|tee log/rcnn/train-from-start-$now.log &  
