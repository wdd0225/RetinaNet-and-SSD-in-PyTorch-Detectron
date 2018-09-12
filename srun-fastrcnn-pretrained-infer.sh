#/usr/bin/env sh


mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

srun --partition=Test --mpi=pmi2 --gres=gpu:1 --job-name=infer_rcnn --kill-on-bad-exit=1 python tools/infer_simple.py  --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml --load_detectron 'data/pretrained_model/faster_rcnn.pkl' --image_dir test --output_dir prediction 2>&1|tee log/rcnn/test-pretrained-infer$now.log &
