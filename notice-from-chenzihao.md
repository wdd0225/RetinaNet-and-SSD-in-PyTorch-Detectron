## Scripts
srun-rcnn-ckpt-infer.sh				load ckpt to infer rcnn
srun-rcnn-pretrained-infer.sh			load pretrained weights to infer rcnn
srun-rcnn-test.sh				load pretrained weights to test rcnn
srun-rcnn-train.sh				train rcnn
srun-retinanet-pretrained-ckpt-train.sh		load ckpt to train retinanet
srun-retinanet-pretrained-detectron-train.sh	load pretrained weights to train retinanet
srun-retinanet-test-ckpt.sh			load ckpt to test retinanet
srun-retinanet-train.sh				train retinanet
srun-ssd-pretrained-ckpt-train.sh		load ckpt to train ssd
srun-ssd-test.sh				load ckpt to test ssd
srun-ssd-train.sh				train ssd

## Folders
best_ckpt					stores the current best mAP ckpt
configs						config files
data						contains coco data and pretrained weights
eval						stores evaluation result of test
lib						codes
log						log files for all scripts
Outputs						stores ckpt for all test, train and infer 
prediction					stores results for infer
pycocotools					package for loading coco data
test						intermediate files generated during test
tools						python scripts directly called for train, test and infer, by the bash scripts
