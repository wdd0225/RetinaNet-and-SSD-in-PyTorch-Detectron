#/usr/bin/env sh
 
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=Test --mpi=pmi2 --gres=gpu:1 --job-name=test_rcnn --kill-on-bad-exit=1 python tools/test_net.py  --dataset coco2014mini --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml --load_detectron data/pretrained_model/faster_rcnn.pkl  2>&1|tee log/rcnn/test-$now.log
