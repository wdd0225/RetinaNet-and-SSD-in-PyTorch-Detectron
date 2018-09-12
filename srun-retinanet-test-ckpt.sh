#/usr/bin/env sh

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
Steps=142499
srun --partition=Test --mpi=pmi2 --gres=gpu:8 --job-name=retinanet_test --kill-on-bad-exit=1 python tools/test_net.py --multi-gpu-testing --dataset coco2014mini --cfg configs/baselines/retinanet_R-50-FPN_1x.yaml --load_ckpt "Outputs/retinanet_R-50-FPN_1x/9w_15w_step/ckpt/model_step$Steps.pth" 2>&1|tee log/retinanet/test-step$Steps-$now.log
