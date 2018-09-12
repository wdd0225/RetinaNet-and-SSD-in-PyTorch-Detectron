from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from PIL import Image
import _init_paths
from lib.core import ssd_config as cfg
from utils.blob import BaseTransform
from datasets.coco_test import COCODetection
from modeling.SSD import build_ssd
import time
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default=cfg.PRETRAINED_WEIGHT,
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--coco_root', default=cfg.COCO_ROOT, help='Location of COCO/VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test_result_new.txt'
    num_images = len(testset)
    print('~~~~~~~~~~~~~~~~~: ', num_images)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(cfg.COCO_CLASSES)+1)]
    for i in range(num_images):
        im, gt, h, w = testset.pull_item(i)

        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        # with open(filename, mode='a') as f:
            # f.write('\nGROUND TRUTH FOR: '+str(img_id)+'\n')
            # for box in annotation:
            #     f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()
        t0 = time.time()
        y = net(x)      # forward pass
        detections = y.data
        # # scale each detection back up to the image
        # scale = torch.Tensor([img.shape[1], img.shape[0],
        #                      img.shape[1], img.shape[0]])
        t1 = time.time()
        print('timer: %.4f sec.' % (t1 - t0),flush=True)
        pred_num = 0
        for j in range(1, detections.size(1)):
            # # if i!=0:
            # j = 0
            # while detections[0, i, j, 0] >= 0.1:
            #     if pred_num == 0:
            #         with open(filename, mode='a') as f:
            #             f.write(str(img_id)+'\n')
            #     score = detections[0, i, j, 0]
            #     label_name = labelmap[i-1]
            #     pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            #     coords = (pt[0], pt[1], pt[2], pt[3])
            #     pred_num += 1
            #     with open(filename, mode='a') as f:
            #         f.write(str(pred_num)+' label: '+str(i)+' score: ' +
            #                 str(score) + ' '.join(str(c) for c in coords) + '\n')
            #     j += 1
            k = 0
            inds = np.where(detections[0, j, k, 0] > 0.01)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets
            k += 1

        print('im_detect: {:d}/{:d}'.format(i + 1, num_images))
    print('Evaluating detections')
    testset.evaluate_detections(all_boxes,save_folder)
    # evaluate_detections(all_boxes, output_dir, test_net)

# def evaluate_detections(box_list, output_dir, dataset):
    # write_voc_results_file(box_list, dataset)
    # do_python_eval(output_dir)



def test_coco():
    # load net
    num_classes = len(cfg.COCO_CLASSES) + 1 # +1 background
    print('num of class:  ', num_classes)
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = COCODetection(args.coco_root)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_coco()
