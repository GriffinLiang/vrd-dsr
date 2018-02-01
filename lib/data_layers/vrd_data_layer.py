import numpy as np
import os.path as osp
import scipy.io as sio
import scipy
import cv2
import cPickle
import sys
sys.path.insert(0, '../')
from lib.blob import prep_im_for_blob
import math


class VrdDataLayer(object):

    def __init__(self, ds_name, stage, model_type = None, proposals_path = None):
        """Setup the RoIDataLayer."""
        self.stage = stage
        self.model_type = model_type
        self.this_dir = osp.dirname(__file__)
        self._classes = [x.strip() for x in open('../data/%s/obj.txt'%ds_name).readlines()]
        self._relations = [x.strip() for x in open('../data/%s/rel.txt'%ds_name).readlines()]
        self._num_classes = len(self._classes)
        self._num_relations = len(self._relations)
        self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
        self._relations_to_ind = dict(zip(self._relations, xrange(self._num_relations)))        
        self._cur = 0
        self.cache_path = '../data/cache'
        with open('../data/%s/%s.pkl'%(ds_name, stage), 'rb') as fid:
            anno = cPickle.load(fid)
        if(self.stage == 'train'):          
            self._anno = [x for x in anno if x is not None and len(x['classes'])>1]
        else:
            self.proposals_path = proposals_path
            if(proposals_path != None):
                with open(proposals_path, 'rb') as fid:   
                    proposals = cPickle.load(fid)
                    self._boxes = proposals['boxes']
                    self._pred_cls = proposals['cls']
                    self._pred_confs = proposals['confs']
            self._anno = anno
        self._num_instance = len(self._anno)
        self._batch_size = 1
        with open('../data/%s/so_prior.pkl'%ds_name, 'rb') as fid:
            self._so_prior = cPickle.load(fid)  

    def forward(self):
        if(self.stage == 'train'):
            return self.forward_train_rank_im()
        else:
            if(self.proposals_path is None):
                return self.forward_test()
            else:
                if(self.model_type == 'LOC'):
                    return self.forward_det_loc()
                else:
                    return self.forward_det()

    def forward_train_rank_im(self):
        """Get blobs and copy them into this layer's top blob vector."""
        anno_img = self._anno[self._cur]
        im_path = anno_img['img_path']
        im = cv2.imread(im_path)
        ih = im.shape[0]    
        iw = im.shape[1]
        PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
        image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
        blob = np.zeros((1,)+image_blob.shape, dtype=np.float32)
        blob[0] = image_blob        
        boxes = np.zeros((anno_img['boxes'].shape[0], 5))        
        boxes[:, 1:5] = anno_img['boxes'] * im_scale
        classes = np.array(anno_img['classes'])
        ix1 = np.array(anno_img['ix1'])
        ix2 = np.array(anno_img['ix2'])
        rel_classes = anno_img['rel_classes']

        n_rel_inst = len(rel_classes)
        rel_boxes = np.zeros((n_rel_inst, 5))
        rel_labels = -1*np.ones((1, n_rel_inst*self._num_relations))
        SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
        rel_so_prior = np.zeros((n_rel_inst, self._num_relations))
        pos_idx = 0
        for ii in range(len(rel_classes)):
            sBBox = anno_img['boxes'][ix1[ii]]
            oBBox = anno_img['boxes'][ix2[ii]]
            rBBox = self._getUnionBBox(sBBox, oBBox, ih, iw)
            rel_boxes[ii, 1:5] = np.array(rBBox) * im_scale    
            SpatialFea[ii] = [self._getDualMask(ih, iw, sBBox), \
                              self._getDualMask(ih, iw, oBBox)]
            rel_so_prior[ii] = self._so_prior[classes[ix1[ii]], classes[ix2[ii]]]
            for r in rel_classes[ii]:
                rel_labels[0, pos_idx] = ii*self._num_relations + r
                pos_idx += 1  
        image_blob = image_blob.astype(np.float32, copy=False)
        boxes = boxes.astype(np.float32, copy=False)
        classes = classes.astype(np.float32, copy=False) 
        self._cur += 1
        if(self._cur >= len(self._anno)):
            self._cur = 0        
        return blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_so_prior

    def forward_test(self):
        """Get blobs and copy them into this layer's top blob vector."""
        anno_img = self._anno[self._cur]
        if(anno_img is None):
            self._cur += 1
            if(self._cur >= len(self._anno)):
                self._cur = 0
            return None
        im_path = anno_img['img_path']
        im = cv2.imread(im_path)
        ih = im.shape[0]    
        iw = im.shape[1]
        PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
        image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
        blob = np.zeros((1,)+image_blob.shape, dtype=np.float32)
        blob[0] = image_blob        
        # Reshape net's input blobs
        boxes = np.zeros((anno_img['boxes'].shape[0], 5))        
        boxes[:, 1:5] = anno_img['boxes'] * im_scale
        classes = np.array(anno_img['classes'])
        ix1 = np.array(anno_img['ix1'])
        ix2 = np.array(anno_img['ix2'])
        rel_classes = anno_img['rel_classes']

        n_rel_inst = len(rel_classes)
        rel_boxes = np.zeros((n_rel_inst, 5))
        SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
        # SpatialFea = np.zeros((n_rel_inst, 8))
        for ii in range(n_rel_inst):
            sBBox = anno_img['boxes'][ix1[ii]]
            oBBox = anno_img['boxes'][ix2[ii]]
            rBBox = self._getUnionBBox(sBBox, oBBox, ih, iw)    
            soMask = [self._getDualMask(ih, iw, sBBox), \
                      self._getDualMask(ih, iw, oBBox)]                        
            rel_boxes[ii, 1:5] = np.array(rBBox) * im_scale
            SpatialFea[ii] = soMask
            # SpatialFea[ii] = self._getRelativeLoc(sBBox, oBBox)
            
        image_blob = image_blob.astype(np.float32, copy=False)
        boxes = boxes.astype(np.float32, copy=False)
        classes = classes.astype(np.float32, copy=False) 
        self._cur += 1
        if(self._cur >= len(self._anno)):
            self._cur = 0                
        return blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, anno_img['boxes']        

    def forward_det(self):
        anno_img = self._anno[self._cur]
        boxes_img = self._boxes[self._cur]
        pred_cls_img = self._pred_cls[self._cur]
        pred_confs_img = self._pred_confs[self._cur]        
        if(boxes_img.shape[0] < 2):
            self._cur += 1
            if(self._cur >= len(self._anno)):
                self._cur = 0
            return None
        im_path = anno_img['img_path']
        im = cv2.imread(im_path)
        ih = im.shape[0]    
        iw = im.shape[1]
        PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
        image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
        blob = np.zeros((1,)+image_blob.shape, dtype=np.float32)
        blob[0] = image_blob        
        # Reshape net's input blobs
        boxes = np.zeros((boxes_img.shape[0], 5))        
        boxes[:, 1:5] = boxes_img * im_scale
        classes = pred_cls_img
        ix1 = []
        ix2 = []
        n_rel_inst = len(pred_cls_img)*(len(pred_cls_img)-1)
        rel_boxes = np.zeros((n_rel_inst, 5))
        SpatialFea = np.zeros((n_rel_inst, 2, 32, 32))
        # SpatialFea = np.zeros((n_rel_inst, 8))
        rel_so_prior = np.zeros((n_rel_inst, self._num_relations))
        i_rel_inst = 0
        for s_idx in range(len(pred_cls_img)):
            for o_idx in range(len(pred_cls_img)):
                if(s_idx == o_idx):
                    continue
                ix1.append(s_idx)
                ix2.append(o_idx)
                sBBox = boxes_img[s_idx]
                oBBox = boxes_img[o_idx]
                rBBox = self._getUnionBBox(sBBox, oBBox, ih, iw)
                soMask = [self._getDualMask(ih, iw, sBBox), \
                      self._getDualMask(ih, iw, oBBox)]                        
                rel_boxes[i_rel_inst, 1:5] = np.array(rBBox) * im_scale
                SpatialFea[i_rel_inst] = soMask
                # SpatialFea[i_rel_inst] = self._getRelativeLoc(sBBox, oBBox)
                rel_so_prior[i_rel_inst] = self._so_prior[classes[s_idx], classes[o_idx]]
                i_rel_inst += 1
        image_blob = image_blob.astype(np.float32, copy=False)
        boxes = boxes.astype(np.float32, copy=False)
        classes = classes.astype(np.float32, copy=False) 
        ix1 = np.array(ix1)
        ix2 = np.array(ix2)
        self._cur += 1
        if(self._cur >= len(self._anno)):
            self._cur = 0
        return blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, boxes_img, pred_confs_img, rel_so_prior

    def forward_det_loc(self):
        anno_img = self._anno[self._cur]
        boxes_img = self._boxes[self._cur]
        pred_cls_img = self._pred_cls[self._cur]
        pred_confs_img = self._pred_confs[self._cur]        
        if(boxes_img.shape[0] < 2):
            self._cur += 1
            if(self._cur >= len(self._anno)):
                self._cur = 0
            return None
        im_path = anno_img['img_path']
        im = cv2.imread(im_path)
        ih = im.shape[0]    
        iw = im.shape[1]
        PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
        image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
        blob = np.zeros((1,)+image_blob.shape, dtype=np.float32)
        blob[0] = image_blob        
        # Reshape net's input blobs
        boxes = np.zeros((boxes_img.shape[0], 5))        
        boxes[:, 1:5] = boxes_img * im_scale
        classes = pred_cls_img
        ix1 = []
        ix2 = []
        n_rel_inst = len(pred_cls_img)*(len(pred_cls_img)-1)
        rel_boxes = np.zeros((n_rel_inst, 5))
        SpatialFea = np.zeros((n_rel_inst, 8))
        rel_so_prior = np.zeros((n_rel_inst, self._num_relations))
        i_rel_inst = 0
        for s_idx in range(len(pred_cls_img)):
            for o_idx in range(len(pred_cls_img)):
                if(s_idx == o_idx):
                    continue
                ix1.append(s_idx)
                ix2.append(o_idx)
                sBBox = boxes_img[s_idx]
                oBBox = boxes_img[o_idx]
                rBBox = self._getUnionBBox(sBBox, oBBox, ih, iw)
                rel_boxes[i_rel_inst, 1:5] = np.array(rBBox) * im_scale
                SpatialFea[i_rel_inst] = self._getRelativeLoc(sBBox, oBBox)
                rel_so_prior[i_rel_inst] = self._so_prior[classes[s_idx], classes[o_idx]]
                i_rel_inst += 1
        image_blob = image_blob.astype(np.float32, copy=False)
        boxes = boxes.astype(np.float32, copy=False)
        classes = classes.astype(np.float32, copy=False) 
        ix1 = np.array(ix1)
        ix2 = np.array(ix2)
        self._cur += 1
        if(self._cur >= len(self._anno)):
            self._cur = 0                
        return blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, boxes_img, pred_confs_img, rel_so_prior

    def _getUnionBBox(self, aBB, bBB, ih, iw, margin = 10):
        return [max(0, min(aBB[0], bBB[0]) - margin), \
            max(0, min(aBB[1], bBB[1]) - margin), \
            min(iw, max(aBB[2], bBB[2]) + margin), \
            min(ih, max(aBB[3], bBB[3]) + margin)]
    
    def _getDualMask(self, ih, iw, bb):
        rh = 32.0 / ih
        rw = 32.0 / iw
        x1 = max(0, int(math.floor(bb[0] * rw)))
        x2 = min(32, int(math.ceil(bb[2] * rw)))
        y1 = max(0, int(math.floor(bb[1] * rh)))
        y2 = min(32, int(math.ceil(bb[3] * rh)))
        mask = np.zeros((32, 32))
        mask[y1 : y2, x1 : x2] = 1
        assert(mask.sum() == (y2 - y1) * (x2 - x1))
        return mask    

    def _getRelativeLoc(self, aBB, bBB):
        sx1, sy1, sx2, sy2 = aBB.astype(np.float32)
        ox1, oy1, ox2, oy2 = bBB.astype(np.float32)
        sw, sh, ow, oh = sx2-sx1, sy2-sy1, ox2-ox1, oy2-oy1
        xy = np.array([(sx1-ox1)/ow, (sy1-oy1)/oh, (ox1-sx1)/sw, (oy1-sy1)/sh])
        wh = np.log(np.array([sw/ow, sh/oh, ow/sw, oh/sh]))
        return np.hstack((xy, wh))

if __name__ == '__main__':    
    pass
