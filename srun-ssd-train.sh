#/usr/bin/bash
partition=Bigvideo #$1
num_gpus=8 #$2
dataset=COCO
now=$(date +"%Y%m%d_%H%M%S")
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun -p ${partition} --mpi=pmi2 --gres=gpu:${num_gpus} -n1 --ntasks-per-node=${num_gpus} --job-name=ssd_train --kill-on-bad-exit=1 \
 python tools/train_multi_gpu.py --dataset $dataset 2>&1 | tee log/ssd/train-$partition-$now.log & 

