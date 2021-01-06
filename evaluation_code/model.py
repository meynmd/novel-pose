# ==========================================================
# Author:  Siddharth Seth
# ==========================================================
"""
Main model file.
"""
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1
import numpy as np

from networks import  pose_encoder
from transformation import transform_3d_2d, get_skeleton_transform_matrix, root_relative_to_view_norm, make_skeleton
from config import opts

def model(inputs, costs_collection='costs'):

        # process input variables
        ret_dict = {}
        img_siz = opts['image_size']
        n_joints = opts['n_joints']

        tgt_im = inputs['future_im']
        pose_3d = inputs['pose_3d']
        

        #add all the inputs to the return dict
        ret_dict['tgt_im'] = tf.reshape(tgt_im, (tf.shape(tgt_im)[0],img_siz,img_siz,3))
        ret_dict['pose_3d'] = tf.reshape(pose_3d, (tf.shape(pose_3d)[0],n_joints,3))


        #choose the gpu based on the number of gpus defined in the config file
        with slim.arg_scope(resnet_v1.resnet_arg_scope()), tf.device("/gpu:1" if len(opts['gpu_ids'])>1 else "/gpu:0"):
            _, inception_dict_tgt = resnet_v1.resnet_v1_50(ret_dict['tgt_im'], is_training=False)


        with tf.device("/gpu:1" if len(opts['gpu_ids'])>1 else "/gpu:0"):
         out_dict_tgt = pose_encoder(inception_dict_tgt['resnet_v1_50/block3/unit_5/bottleneck_v1'], reuse=tf.AUTO_REUSE)# 28x28x512

        ret_dict['tgt_out_cam_angles'] = out_dict_tgt['out_cam_angles']
        ret_dict['tgt_out_cam_params'] = out_dict_tgt['out_cam_params']
        ret_dict['tgt_pred_ske'] = out_dict_tgt['pred_ske']


        batch_zeros = tf.zeros_like(out_dict_tgt['out_cam_angles'][:, 0:1])
        trans = 20.0
        out_dict_tgt['trans'] = tf.reshape(tf.stack([batch_zeros, batch_zeros+trans, batch_zeros], 1), (-1,3,1))
        out_dict = transform_3d_2d(out_dict_tgt)
        ret_dict['tgt_trans'] = out_dict_tgt['trans']
        ret_dict['tgt_rot_ske'] = out_dict['rot_ske']
        ret_dict['tgt_projs_skeleton_2d'] = out_dict['projs_skeleton_2d']
        ret_dict['tgt_unscaled_projs_skeleton_2d'] = out_dict['unscaled_projs_skeleton_2d']
        ret_dict['tgt_focal_length'] = out_dict['focal_length']
        
        return ret_dict
