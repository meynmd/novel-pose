import os.path as osp
import os
import cv2
import numpy as np
import random
import copy
import scipy.io as sio
import tensorflow as tf
import glob as glob
import json
from natsort import natsorted
import math
import imutils
import pickle

from config import opts
from transform_util import root_relative_to_view_norm_skeleton


def get_feed_dict(args, i):
    bs = opts["batch_size"]
    src_im = cv2.resize(cv2.imread(args.src_im), (224,224))
    tgt_im = cv2.resize(cv2.imread(args.tgt_im), (224,224))
    tgt_im = (tgt_im[:,:,::-1] - np.array([103.939, 116.779, 123.68]))/255.
    src_im = src_im[:,:,::-1] /255.
    return {
	i["source_im"] : np.stack([src_im]*bs, 0), 
	i["future_im"] :  np.stack([tgt_im]*bs, 0), 
	i["pose_src_centered"] : np.stack([pose_src_centered]*bs, 0),
	i["pose_src" ] :  np.stack([pose_src]*bs, 0),
	i["parts_src_centered"] :  np.stack([parts_src_centered]*bs, 0),
	i["unc_parts_src_centered"] :  np.stack([unc_parts_src_centered]*bs, 0)
    }
    
def get_padded_part(part, part_mid, pose):
    """
    take the part to the center of a map and pad
    with zeros around it - for every individual part
    Args:
        part - the part map
        part_mid - part center coordinates

    """
    # get the bounding box around the part
    y1, y2 = np.where(part>0)[0][0], np.where(part>0)[0][-1]
    x1, x2 = np.where(part.T>0)[1][0], np.where(part.T>0)[1][-1]
    x_mid, y_mid = part_mid[0], part_mid[1]
    total_pad_x = 112 - (x2 - x1)
    total_pad_y = 112 - (y2 - y1)

    part_secluded = part[y1:y2, x1:x2]

    pad_left, pad_right = 56 - (x_mid-x1), 56 - (x2-x_mid)
    pad_top, pad_bottom = 56 - (y_mid-y1), 56 - (y2-y_mid)

    part_secluded_padded = cv2.copyMakeBorder(part_secluded, int(pad_top), int(pad_bottom), int(pad_left), int(pad_right), cv2.BORDER_CONSTANT)

    return part_secluded, [pose[0]-x1, pose[1]-y1, pose[2]-x1, pose[3]-y1], part_secluded_padded, \
            [pose[0]-x1+pad_left, pose[1]-y1+pad_top, pose[2]-x1+pad_left, pose[3]-y1+pad_top]

#part deformation model related globals
info1 = sio.loadmat('./no_occ.mat')
image1 = cv2.resize(info1['image'], (112, 112))
pose1 = info1['pose_2d']/2
part_segments1 = info1['part_segments']
limbs = np.array([[0,1], [9,10], [10,11], [11,12], [13,14], [14,15], [15,16], [1,8], [2,3], [3,4], [5,6], [6,7]])
parts_src = []

for l in range(len(limbs)):
    parts_src.append(np.stack([part_segments1[:,:,l],part_segments1[:,:,l],part_segments1[:,:,l]],2))
parts_src = np.array(parts_src)
parts_src_rgb = np.sum(parts_src,0)

   
p = [] # width scale variable
q = [] # length scale variable
limb_centers_parts_src = []
parts_src_centered = []
pose_src_centered = []
pose_src = []

for i,l in enumerate(limbs):
    x1, y1 = pose1[l[0]][0], pose1[l[0]][1]
    x2, y2 = pose1[l[1]][0], pose1[l[1]][1]
    x_mid, y_mid = (x1+x2)//2, (y1+y2)//2
    
    p.append(1)
    # q.append(get_dist(x1_1, y1_1, x2_1, y2_1) / get_dist(x1, y1, x2, y2))
    pose_src.append(np.array([x1, y1, x2, y2]))
    limb_centers_parts_src.append(np.array([x_mid, y_mid]))
    
    c, d, a, b = get_padded_part(parts_src[i], limb_centers_parts_src[i], [x1,y1,x2,y2])
    parts_src_centered.append(a)
    pose_src_centered.append(b)

p = np.array(p)
# q = np.array(q)
limb_centers_parts_src = np.array(limb_centers_parts_src)
pose_src_centered = np.array(pose_src_centered).astype(np.float32)
parts_src_centered = np.array(parts_src_centered)       # 12, 112, 112
pose_src = np.array(pose_src).astype(np.float32)
parts_src_centered_t = np.transpose(parts_src_centered[:,:,:,0], (1, 2, 0))
eroded_mask = cv2.erode((parts_src_centered_t*255).astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
dilated_mask = cv2.dilate((parts_src_centered_t*255).astype(np.uint8), np.ones((2,2), np.uint8), iterations=1)
eroded_blurred_mask = cv2.GaussianBlur(dilated_mask[:,:,:12]-eroded_mask[:,:,:12],(3,3),0)
unc_parts_src_centered = np.transpose(eroded_blurred_mask, (2, 0, 1))
unc_parts_src_centered = (np.stack([unc_parts_src_centered, unc_parts_src_centered, unc_parts_src_centered], 3)/255.).astype(np.float32)

eroded_mask = cv2.erode((parts_src_centered_t*255).astype(np.uint8), np.ones((2,2), np.uint8), iterations=1)
eroded_blurred_mask = cv2.GaussianBlur(eroded_mask,(3,3),0)
parts_src_centered_t = eroded_blurred_mask

#dilate the parts 2 more iterations using a 3x3 kernel
parts_src_centered_t = cv2.dilate(parts_src_centered_t.astype(np.uint8),  np.ones((3,3), np.uint8),iterations = 1)

parts_src_centered = np.transpose(parts_src_centered_t, (2, 0, 1))
parts_src_centered = (np.stack([parts_src_centered, parts_src_centered, parts_src_centered], 3)/255.).astype(np.float32)


