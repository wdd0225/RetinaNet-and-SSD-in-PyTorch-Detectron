#!/usr/bin/env sh
 
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=Bigvideo --mpi=pmi2 --gres=gpu:8 --job-name=retinanet_train --kill-on-bad-exit=1 python tools/train_net_step.py --resume --nw 4 --load_ckpt 'Outputs/retinanet_R-50-FPN_1x/9w_15w_step/ckp/model_step142499.pth' --dataset coco2017 --cfg configs/baselines/retinanet_R-50-FPN_1x.yaml  --use_tfboard --bs 16 --iter_size 1 2>&1|tee log/retinanet/train-pretrained-ckpt-$now.log &  
