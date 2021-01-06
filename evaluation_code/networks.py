"""
Network definitions
"""

import tensorflow as tf
from transformation import unit_norm_tf, make_skeleton
from config import opts
import numpy as np


############################
#Helpers for Pose_encoder
############################

#Fully connected residual block with 2 FC layers
def res_block(inp, units, alpha, block_id, reuse, training = False):
    with tf.variable_scope(block_id, reuse = reuse):
        fc = tf.layers.dense(inp, units, activation = None, name = "fc1")
        bn = tf.layers.batch_normalization(fc, training = training, fused = True)
        lr = tf.maximum(alpha*bn, bn)

        fc2 = tf.layers.dense(lr, units, activation  = None, name = "fc2")
        bn2 = tf.layers.batch_normalization(fc2, training = training, fused = True)
        lr2 = tf.maximum(alpha*bn2, bn2)

    return inp + lr2

#returns camera angles and params, from the given tensor "inp"
def cam_params_angles(inp, name, reuse):
    with tf.variable_scope(name, reuse = reuse):
        fc_embedding = tf.layers.dense(inp, 6, activation=None, name='fc_embedding')
        fc_embedding = tf.nn.tanh(fc_embedding/10.0)
        fc_sin = tf.slice(fc_embedding, [0, 0], [-1, 3], name='layer_sine')
        fc_cos = tf.slice(fc_embedding, [0, 3], [-1, 3], name='layer_cosine')
        deno = tf.sqrt(tf.add(tf.square(fc_sin),tf.square(fc_cos)))
        fc_sin_mod = tf.div(fc_sin,deno)
        fc_cos_mod = tf.div(fc_cos,deno)
        fc_embed_atan2 = tf.atan2(fc_sin_mod, fc_cos_mod, name='layer_atan2')

        fc_cam_params = tf.layers.dense(inp, 2, activation=None, name='fc_cam_params')

    return fc_embed_atan2, fc_cam_params

#returns predicted skeleton (J, 3), from the given tensor "inp"
def get_pred_ske(inp, name, reuse):
    with tf.variable_scope(name, reuse = reuse):
        out_pose = tf.layers.dense(inp, (opts['n_joints'] - 4) * 3 + 1, activation=None, name='pose')
        out_pose_reshaped = tf.reshape(out_pose[:, :(opts['n_joints'] - 4) * 3], (-1, opts['n_joints'] - 4, 3))

        out_angle = out_pose[:, (opts['n_joints'] - 4) * 3:]

        out_ske = make_skeleton(out_pose_reshaped, out_angle)

    return out_ske

def pose_encoder(img_features, reuse=False):
    batch_norm = False
    units = 1024  #number of units in each FC layer of the encoder network
    ret_dict = {}
    alpha = 0.1

    with tf.variable_scope('pose_encoder', reuse=reuse):
        with tf.variable_scope('com_branch', reuse=reuse):
            flatten = tf.reshape(img_features, (-1, img_features.shape[1]*img_features.shape[2]*img_features.shape[3]))     # Bx14x14x512

            fc_1 = tf.layers.dense(flatten, units, activation=None, name='fc_1')
            bn1 = tf.layers.batch_normalization(fc_1, training=batch_norm, fused=True)
            lr1 = tf.maximum(alpha*bn1, bn1)

            fc_2 = tf.layers.dense(lr1, units, activation=None, name='fc_2')
            bn2 = tf.layers.batch_normalization(fc_2, training=batch_norm, fused=True)
            lr2 = tf.maximum(alpha*bn2, bn2)
            fc_3 = tf.layers.dense(lr2, units, activation=None, name='fc_3')
            bn3 = tf.layers.batch_normalization(fc_3, training=batch_norm, fused=True)
            lr3 = tf.maximum(alpha*bn3, bn3)

        
        with tf.variable_scope("branch_2", reuse = reuse):
            with tf.variable_scope("cam_angles2", reuse = reuse):
                res_block_5 = res_block(lr3, units, alpha, "res_block_5", reuse)
                res_block_6 = res_block(res_block_5, units, alpha, "res_block_6", reuse)

                res_block_7 = res_block(res_block_6, units, alpha, "res_block_7", reuse)
                res_block_8 = res_block(res_block_7, units, alpha, "res_block_8", reuse)

                fc_embed_atan2, fc_cam_params = cam_params_angles(res_block_8, "cam_a3", reuse)

                ret_dict['out_cam_angles'] = fc_embed_atan2
                ret_dict['out_cam_params'] = fc_cam_params

            with tf.variable_scope("cam_params2", reuse = reuse):
                res_block_9 = res_block(lr3, units, alpha, "res_block_9", reuse)
                res_block_10 = res_block(res_block_9, units, alpha, "res_block_10", reuse)

                res_block_11 = res_block(res_block_10, units, alpha, "res_block_11", reuse)
                res_block_12 = res_block(res_block_11, units, alpha, "res_block_12", reuse)

                out_ske = get_pred_ske(res_block_12, "cam_p3", reuse)

                ret_dict['pred_ske'] = out_ske

    return ret_dict



