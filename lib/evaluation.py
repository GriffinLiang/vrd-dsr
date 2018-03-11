import scipy.io as sio
import numpy as np
import cPickle
import copy
import time
import sys
import os.path as osp
this_dir = osp.dirname(osp.realpath(__file__))
print this_dir

def eval_per_image(i, gt, pred, use_rel, gt_thr = 0.5, return_match = False):
    gt_tupLabel = gt['tuple_label'][i].astype(np.float32)
    num_gt_tuple = gt_tupLabel.shape[0]
    if(num_gt_tuple == 0 or pred['tuple_confs'][i] is None):
        return 0, 0
    if(not use_rel):    
        gt_tupLabel = gt_tupLabel[:, (0,2)]
    gt_objBox = gt['obj_bboxes'][i].astype(np.float32)
    gt_subBox = gt['sub_bboxes'][i].astype(np.float32)
    
    gt_detected = np.zeros((num_gt_tuple, 1), np.float32)       
    labels = pred['tuple_label'][i].astype(np.float32)
    boxSub = pred['sub_bboxes'][i].astype(np.float32)
    boxObj = pred['obj_bboxes'][i].astype(np.float32)
    num_tuple = labels.shape[0]

    tp = np.zeros((num_tuple,1))
    fp = np.zeros((num_tuple,1))
    for j in range(num_tuple):
        bbO = boxObj[j,:]
        bbS = boxSub[j,:]
        ovmax = gt_thr
        kmax = -1
        for k in range(num_gt_tuple):
            if np.linalg.norm(labels[j,:] - gt_tupLabel[k,:],2) != 0:
                continue;                
            if gt_detected[k] > 0:
                continue;
            # [xmin, ymin, xmax, ymax] 
            bbgtO = gt_objBox[k,:];
            bbgtS = gt_subBox[k,:];
            
            biO = np.array([max(bbO[0],bbgtO[0]), max(bbO[1],bbgtO[1]), min(bbO[2],bbgtO[2]), min(bbO[3],bbgtO[3])])
            iwO=biO[2]-biO[0]+1;
            ihO=biO[3]-biO[1]+1;
        
            biS = np.array([max(bbS[0],bbgtS[0]), max(bbS[1],bbgtS[1]), min(bbS[2],bbgtS[2]), min(bbS[3],bbgtS[3])])
            iwS=biS[2]-biS[0]+1;
            ihS=biS[3]-biS[1]+1;
            
            if iwO>0 and ihO>0 and iwS>0 and ihS>0:
                # compute overlap as area of intersection / area of union
                uaO=(bbO[2]-bbO[0]+1)*(bbO[3]-bbO[1]+1) + \
                    (bbgtO[2]-bbgtO[0]+1)*(bbgtO[3]-bbgtO[1]+1) - iwO*ihO;
                ovO =iwO*ihO/uaO;
                
                uaS=(bbS[2]-bbS[0]+1)*(bbS[3]-bbS[1]+1) + \
                    (bbgtS[2]-bbgtS[0]+1)*(bbgtS[3]-bbgtS[1]+1) - iwS*ihS
                ovS =iwS*ihS/uaS
                ov = min(ovO, ovS)
                
                # makes sure that this object is detected according
                # to its individual threshold
                if ov >= ovmax:
                    ovmax=ov;
                    kmax=k;
        if kmax > -1:
            tp[j] = 1;
            gt_detected[kmax] = 1;
        else:
            fp[j] = 1;
    if(return_match):
        return tp
    return tp.sum(), num_gt_tuple

def eval_reall_at_N(ds_name, N, res, use_rel = True, use_zero_shot = False):
    if(ds_name == 'vrd'):
        num_imgs = 1000
        gt = sio.loadmat('../data/vrd/gt.mat')
        gt['tuple_label'] = gt['gt_tuple_label'][0]
        gt['obj_bboxes'] = gt['gt_obj_bboxes'][0]
        gt['sub_bboxes'] = gt['gt_sub_bboxes'][0]
        if(use_zero_shot):
            zs = sio.loadmat('../data/vrd/zeroShot.mat')['zeroShot'][0];
            for ii in range(num_imgs):
                if(zs[ii].shape[0] == 0):
                    continue
                idx = zs[ii] == 1                
                gt['tuple_label'][ii] = gt['tuple_label'][ii][idx[0]]
                gt['obj_bboxes'][ii] = gt['obj_bboxes'][ii][idx[0]]
                gt['sub_bboxes'][ii] = gt['sub_bboxes'][ii][idx[0]]
    else:
        # Testing all images is quite slow.
        num_imgs = 8995         
        if(use_zero_shot):
            gt_path = '../data/%s/zs_gt.pkl'%ds_name
        else:
            gt_path = '../data/%s/gt.pkl'%ds_name
        with open(gt_path, 'rb') as fid:
            gt = cPickle.load(fid)
        
    pred = {}
    pred['tuple_label'] = copy.deepcopy(res['rlp_labels_ours'])
    pred['tuple_confs'] = copy.deepcopy(res['rlp_confs_ours'])
    pred['sub_bboxes']  = copy.deepcopy(res['sub_bboxes_ours'])
    pred['obj_bboxes']  = copy.deepcopy(res['obj_bboxes_ours'])

    for ii in range(num_imgs):
        if(pred['tuple_confs'][ii] is None):
            continue        
        pred['tuple_confs'][ii] = np.array(pred['tuple_confs'][ii])
        if(pred['tuple_confs'][ii].shape[0] == 0):
            continue        
        idx_order = np.array(pred['tuple_confs'][ii]).argsort()[::-1][:N]        
        pred['tuple_label'][ii] = pred['tuple_label'][ii][idx_order,:]
        pred['tuple_confs'][ii] = pred['tuple_confs'][ii][idx_order]
        pred['sub_bboxes'][ii]  = pred['sub_bboxes'][ii][idx_order,:]
        pred['obj_bboxes'][ii]  = pred['obj_bboxes'][ii][idx_order,:]        
    # from IPython import embed; embed()
    tp_num = 0
    num_pos_tuple = 0    
    for i in range(num_imgs):
        img_tp, img_gt = eval_per_image(i, gt, pred, use_rel, gt_thr = 0.5)
        tp_num += img_tp
        num_pos_tuple += img_gt
    recall = (tp_num/num_pos_tuple)
    return recall*100        

def eval_obj_img(gt_boxes, gt_cls, pred_boxes, pred_cls, gt_thr=0.5, return_flag = False):
    pos_num = 0;
    loc_num = 0;
    dets = []
    gts = []
    for ii in range(pred_boxes.shape[0]):
        recog_flag = -1;
        gt_flag = -1;
        bbox = pred_boxes[ii]
        for jj in range(gt_boxes.shape[0]):
            gt_box = gt_boxes[jj]
            
            in_box = np.array([max(bbox[0],gt_box[0]), max(bbox[1],gt_box[1]), min(bbox[2],gt_box[2]), min(bbox[3],gt_box[3])])
            in_box_w = in_box[2]-in_box[0]+1;
            in_box_h = in_box[3]-in_box[1]+1;
            
            if in_box_w>0 and in_box_h>0:
                # compute overlap as area of intersection / area of union
                un_box_area = 1.0*(bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1) + \
                             (gt_box[2]-gt_box[0]+1)*(gt_box[3]-gt_box[1]+1) - in_box_w*in_box_h;
                IoU = 1.0*in_box_w*in_box_h/un_box_area;
                # makes sure that this object is detected according
                # to its individual threshold
                if IoU >= gt_thr:
                    gt_flag = gt_cls[jj]
                    if np.linalg.norm(gt_cls[jj] - pred_cls[ii]) == 0:
                        recog_flag = 1                        
                        break
                    else:
                        recog_flag = 2
        if(recog_flag == 1):
            pos_num += 1
        elif(recog_flag == 2):
            loc_num += 1
        dets.append(recog_flag)
        gts.append(gt_flag)
    if(return_flag):
        return dets, gts        
    return pos_num, loc_num

def eval_object_recognition_top_N(proposals_path):    
    with open('VRD_test.pkl', 'rb') as fid:
        anno = cPickle.load(fid)            
    
    with open(proposals_path, 'rb') as fid:   
        proposals = cPickle.load(fid)

    pos_num = 0.0
    loc_num = 0.0
    for ii in range(len(anno)):
        if(anno[ii] is None):
            continue
        anno_img = anno[ii]
        gt_boxes = anno_img['boxes'].astype(np.float32)
        gt_cls = anno_img['classes'].astype(np.float32)
        pred_boxes = proposals['boxes'][ii].astype(np.float32)
        pred_cls = proposals['cls'][ii].astype(np.float32)
        pos_num_img, loc_num_img = eval_obj_img(gt_boxes, gt_cls, pred_boxes, pred_cls)
        pos_num += pos_num_img
        loc_num += loc_num_img
    print pos_num/(pos_num+loc_num)

if __name__ == '__main__':
    pass
