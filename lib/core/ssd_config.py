# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

SAVE_CKPT = 20000

PRETRAINED_WEIGHT = '/mnt/lustre/chenzihao/mask-rcnn.pytorch/Outputs/ssd300_COCO/399999.pth'
# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
VOC_ROOT = '/mnt/lustre/chenzihao/mask-rcnn.pytorch/data/coco'
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
VOC_ON = False #Not used yet

coco = {
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

COCO_ROOT = '/mnt/lustre/chenzihao/mask-rcnn.pytorch/data/coco'
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
LOG_ITER = 10
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')
COCO_ON = False #Not used yet
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}
COCO_LABEL_MAP = [
[1,1,"person"],
[2,2,"bicycle"],
[3,3,"car"],
[4,4,"motorcycle"],
[5,5,"airplane"],
[6,6,"bus"],
[7,7,"train"],
[8,8,"truck"],
[9,9,"boat"],
[10,10,"traffic light"],
[11,11,"fire hydrant"],
[13,12,"stop sign"],
[14,13,"parking meter"],
[15,14,"bench"],
[16,15,"bird"],
[17,16,"cat"],
[18,17,"dog"],
[19,18,"horse"],
[20,19,"sheep"],
[21,20,"cow"],
[22,21,"elephant"],
[23,22,"bear"],
[24,23,"zebra"],
[25,24,"giraffev"],
[27,25,"backpack"],
[28,26,"umbrella"],
[31,27,"handbag"],
[32,28,"tie"],
[33,29,"suitcase"],
[34,30,"frisbeev"],
[35,31,"skis"],
[36,32,"snowboard"],
[37,33,"sports ball"],
[38,34,"kite"],
[39,35,"baseball bat"],
[40,36,"baseball glove"],
[41,37,"skateboard"],
[42,38,"surfboard"],
[43,39,"tennis racket"],
[44,40,"bottle"],
[46,41,"wine glass"],
[47,42,"cup"],
[48,43,"fork"],
[49,44,"knife"],
[50,45,"spoon"],
[51,46,"bowl"],
[52,47,"banana"],
[53,48,"apple"],
[54,49,"sandwich"],
[55,50,"orange"],
[56,51,"broccoli"],
[57,52,"carrot"],
[58,53,"hot dog"],
[59,54,"pizza"],
[60,55,"donut"],
[61,56,"cake"],
[62,57,"chair"],
[63,58,"couch"],
[64,59,"potted plant"],
[65,60,"bed"],
[67,61,"dining table"],
[70,62,"toilet"],
[72,63,"tv"],
[73,64,"laptop"],
[74,65,"mouse"],
[75,66,"remote"],
[76,67,"keyboard"],
[77,68,"cell phone"],
[78,69,"microwave"],
[79,70,"oven"],
[80,71,"toaster"],
[81,72,"sink"],
[82,73,"refrigerator"],
[84,74,"book"],
[85,75,"clock"],
[86,76,"vase"],
[87,77,"scissors"],
[88,78,"teddy bearv"],
[89,79,"hair drier"],
[90,80,"toothbrush"]]
