# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, 'faster-rcnn/lib'))
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, res_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from easydict import EasyDict

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class detector():

  def __init__(self):
    self.args = EasyDict()
    self.args.dataset = 'vrd'
    self.args.cfg_file = 'faster-rcnn/cfgs/vgg16.yml'
    self.args.net = 'vgg16'
    self.args.load_dir = 'models/faster_rcnn_1_20_7559.pth'
    self.args.cuda = True
    self.args.mGPUs = False
    self.args.class_agnostic = False
    self.args.parallel_type = 0
    self.args.batch_size = 1
    print(self.args)

    if self.args.cfg_file is not None:
      cfg_from_file(self.args.cfg_file)
    
    cfg.USE_GPU_NMS = self.args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    load_name = self.args.load_dir  
    with open('data/vrd/obj.txt') as f:
      self.vrd_classes = [ x.strip() for x in f.readlines() ]
    self.vrd_classes = ['__background__'] + self.vrd_classes

    self.fasterRCNN = vgg16(self.vrd_classes, pretrained=False, class_agnostic=self.args.class_agnostic)
    self.fasterRCNN.create_architecture()

    if self.args.cuda > 0:
      checkpoint = torch.load(load_name)
    else:
      checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    self.fasterRCNN.load_state_dict(checkpoint['model'])

    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    print("load checkpoint %s" % (load_name))

    # initilize the tensor holder here.
    self.im_data = torch.FloatTensor(1)
    self.im_info = torch.FloatTensor(1)
    self.num_boxes = torch.LongTensor(1)
    self.gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if self.args.cuda > 0:
      self.im_data = self.im_data.cuda()
      self.im_info = self.im_info.cuda()
      self.num_boxes = self.num_boxes.cuda()
      self.gt_boxes = self.gt_boxes.cuda()

    # make variable
    self.im_data = Variable(self.im_data, volatile=True)
    self.im_info = Variable(self.im_info, volatile=True)
    self.num_boxes = Variable(self.num_boxes, volatile=True)
    self.gt_boxes = Variable(self.gt_boxes, volatile=True)

    if self.args.cuda > 0:
      cfg.CUDA = True

    if self.args.cuda > 0:
      self.fasterRCNN.cuda()

    self.fasterRCNN.eval()       

  def _get_image_blob(self, im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
      im_scale = float(target_size) / float(im_size_min)
      # Prevent the biggest axis from being more than MAX_SIZE
      if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
      im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
              interpolation=cv2.INTER_LINEAR)
      im_scale_factors.append(im_scale)
      processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

# def detect_im():
# if __name__ == '__main__':

  def det_im(self, im_file):
    max_per_image = 100
    thresh = 0.05 
    total_tic = time.time()    
    # im = cv2.imread(im_file)
    im_in = np.array(imread(im_file))
    if len(im_in.shape) == 2:
      im_in = im_in[:,:,np.newaxis]
      im_in = np.concatenate((im_in,im_in,im_in), axis=2)
    # rgb -> bgr
    im = im_in[:,:,::-1]

    blobs, im_scales = self._get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    self.im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    self.im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
    self.gt_boxes.data.resize_(1, 1, 5).zero_()
    self.num_boxes.data.resize_(1).zero_()
    
    det_tic = time.time()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          if self.args.class_agnostic:
              if self.args.cuda > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

              box_deltas = box_deltas.view(1, -1, 4)
          else:
              if self.args.cuda > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
              box_deltas = box_deltas.view(1, -1, 4 * len(self.vrd_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()
    im2show = np.copy(im)
    res = {}
    res['box'] = np.zeros((0,4))
    res['cls'] = []
    res['confs'] = []
    for j in xrange(1, len(self.vrd_classes)):
      inds = torch.nonzero(scores[:,j]>thresh).view(-1)
      # if there is det
      if inds.numel() > 0:
        cls_scores = scores[:,j][inds]
        _, order = torch.sort(cls_scores, 0, True)
        if self.args.class_agnostic:
          cls_boxes = pred_boxes[inds, :]
        else:
          cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
        
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets = cls_dets[order]
        keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
        cls_dets = cls_dets[keep.view(-1).long()]        
        im2show = res_detections(im2show, j, self.vrd_classes[j], cls_dets.cpu().numpy(), res, 0.5)

    misc_toc = time.time()
    nms_time = misc_toc - misc_tic

    sys.stdout.write('im_detect: {:.3f}s {:.3f}s   \r'.format(detect_time, nms_time))
    sys.stdout.flush()
    cv2.imwrite('img/im_det.jpg', im2show)
    return res

if __name__ == '__main__':
  det = detector()
  det.det_im('img/im.jpg')
