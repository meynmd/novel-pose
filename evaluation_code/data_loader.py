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

from config import opts

from h36_load_annots import annot_files, frames, cameras
from transform_util import root_relative_to_view_norm_skeleton

#h36_datapath
h36_datapath = './H36_cropped'
subjects = ['S9','S11']
activities = os.listdir('./H36_cropped/S11')

def unit_norm(mat, dim=1):
    norm = (np.sqrt(np.sum(mat ** 2, dim)) + 1e-9)
    norm = np.expand_dims(norm, dim)
    mat = mat / norm
    return mat

def normalize_3d_pose(pred_3d):
    pelvis = pred_3d[0,:]
    rhip = unit_norm(pred_3d[9,:], dim=0) * 1.05 + pelvis
    lhip = unit_norm(pred_3d[13,:], dim=0) * 1.05 + pelvis
    neck = unit_norm(pred_3d[1,:], dim=0) * 4.75 + pelvis
    rs = unit_norm(pred_3d[2,:]-pred_3d[1,:], dim=0) * 1.37 + neck
    re = unit_norm(pred_3d[3,:]-pred_3d[2,:], dim=0) * 2.8 + rs
    rh = unit_norm(pred_3d[4,:]-pred_3d[3,:], dim=0) * 2.4 + re
    ls = unit_norm(pred_3d[5,:]-pred_3d[1,:], dim=0) * 1.37 + neck
    le = unit_norm(pred_3d[6,:]-pred_3d[5,:], dim=0) * 2.8 + ls
    lh = unit_norm(pred_3d[7,:]-pred_3d[6,:], dim=0) * 2.4 + le
    head = unit_norm(pred_3d[8,:]-pred_3d[1,:], dim=0) * 2.0 + neck
    rk = unit_norm(pred_3d[10,:]-pred_3d[9,:], dim=0) * 4.2 + rhip
    ra = unit_norm(pred_3d[11,:]-pred_3d[10,:], dim=0) * 3.6 + rk
    rf = unit_norm(pred_3d[12,:]-pred_3d[11,:], dim=0) *2.0 + ra
    lk = unit_norm(pred_3d[14,:]-pred_3d[13,:], dim=0) * 4.2 + lhip
    la = unit_norm(pred_3d[15,:]-pred_3d[14,:], dim=0) * 3.6 + lk
    lf = unit_norm(pred_3d[16,:]-pred_3d[15,:], dim=0) *2.0 + la

    skeleton = np.concatenate([np.expand_dims(pelvis, axis=0), np.expand_dims(neck, axis=0), 
                           np.expand_dims(rs, axis=0), np.expand_dims(re, axis=0),
                           np.expand_dims(rh, axis=0), np.expand_dims(ls, axis=0),
                           np.expand_dims(le, axis=0), np.expand_dims(lh, axis=0),
                           np.expand_dims(head, axis=0), np.expand_dims(rhip, axis=0),
                           np.expand_dims(rk, axis=0), np.expand_dims(ra, axis=0),
                           np.expand_dims(rf, axis=0), np.expand_dims(lhip, axis=0), 
                           np.expand_dims(lk, axis=0), np.expand_dims(la, axis=0), 
                           np.expand_dims(lf, axis=0)], axis=0)
    return skeleton


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

def get_test_set():
    data_list = []
    for subject in ["S11", "S9"]:
        for activity in frames[subject]:
            frameslist = list(set(frames[subject][activity]))
            #assert len(frameslist) == len(set(frameslist))
            for frameid in range(0, len(frameslist)):
                    camera = "58860488"
                    frame = frameslist[frameid]
                    newent = "%s %s %s %s"%(subject, activity, frame, camera)
                    data_list.append(newent)
    print("The length of the test set is", len(data_list), len(data_list[:opts['batch_size']*(len(data_list)//opts['batch_size'])]))
    return data_list[:opts['batch_size']*(len(data_list)//opts['batch_size'])]  #drop remainder to make sure all batches have the same size

def read_data(s):
    """ data reader for test set"""

    subject, activity, frame, camera = s.split(" ")
    frame = int(frame)
    im_file_name = 'img_' + (6 - len(str(frame)))*'0' + str(frame) + '.jpg'
    im_path = os.path.join(h36_datapath, os.path.join(subject + '/' + activity + '/imageSequence/' + str(camera), im_file_name))
    person_image = cv2.imread(im_path)[:,:,::-1]
    person_image = ((person_image - np.array([103.939, 116.779, 123.68]))/255.).astype(np.float32)

    frame = list(frames[subject][activity]).index(frame)

    pose_3d_32 = annot_files[subject][activity]['poses3d'][frame]
    
    pose_3d = np.stack([pose_3d_32[0,:],pose_3d_32[16,:],pose_3d_32[25,:],pose_3d_32[26,:],pose_3d_32[27,:],pose_3d_32[17,:],pose_3d_32[18,:],pose_3d_32[19,:],pose_3d_32[15,:],pose_3d_32[1,:],pose_3d_32[2,:],pose_3d_32[3,:],pose_3d_32[4,:],pose_3d_32[6,:],pose_3d_32[7,:],pose_3d_32[8,:],pose_3d_32[10,:]],0)
    pose_3d[0,:] = (pose_3d[9,:] + pose_3d[13,:])/2.
    pose_3d = pose_3d - pose_3d[0]

    pose_3d_final = normalize_3d_pose(pose_3d)

    pose_3d = root_relative_to_view_norm_skeleton(pose_3d_final)[1].astype(np.float32)

    return person_image, pose_3d




