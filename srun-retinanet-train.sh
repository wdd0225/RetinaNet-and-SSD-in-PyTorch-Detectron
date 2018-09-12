#!/usr/bin/env sh
 
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=TITANXP --mpi=pmi2 --gres=gpu:8 --job-name=RetinaNet --kill-on-bad-exit=1 python tools/train_net_step.py --nw 4 --dataset coco2017 --cfg configs/baselines/retinanet_R-50-FPN_1x.yaml  --use_tfboard --bs 16 --iter_size 1 2>&1|tee log/retinanet/train-from-start-$now.log &  
