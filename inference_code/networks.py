"""
Network definitions
"""

import tensorflow as tf
from transformation import unit_norm_tf, make_skeleton
from config import opts
import numpy as np




def app_encoder(app_img_features, reuse=False):
    alpha = 0.1
    batch_norm = True
    with tf.variable_scope('app_encoder', reuse=reuse):
        
        app_img_features = tf.reshape(app_img_features, (-1, app_img_features.shape[1], app_img_features.shape[2], app_img_features.shape[3]))
        conv1 = tf.layers.conv2d(app_img_features, 1024, 3, strides=2, padding='SAME', \
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn1 = tf.layers.batch_normalization(conv1, training=batch_norm, fused=True)
        relu1 = tf.nn.relu(bn1)        # 14x14x1024

        conv2 = tf.layers.conv2d(relu1, 2048, 3, strides=2, padding='SAME', \
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn2 = tf.layers.batch_normalization(conv2, training=batch_norm, fused=True)
        relu2 = tf.nn.relu(bn2)        # 7x7x2048

        conv3 = tf.layers.conv2d(relu2, 4096, 3, strides=2, padding='SAME', \
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn3 = tf.layers.batch_normalization(conv3, training=batch_norm, fused=True)
        relu3 = tf.nn.relu(bn3)        # 4x4x4096
        

        gap = tf.reduce_mean(relu3, axis=[1,2], keepdims=True)   # 1x1x4096

        gap_reshaped = tf.reshape(app_img_features, (-1, app_img_features.shape[1], app_img_features.shape[2], app_img_features.shape[3]))

        
        conv4 = tf.layers.conv2d(gap_reshaped, 256, 3, strides=1, padding='SAME', 
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn4 = tf.layers.batch_normalization(conv4, training=batch_norm, fused=True)
        relu4 = tf.nn.relu(bn4)        # 4x4x4096

        conv5 = tf.layers.conv2d(relu4, 256, 3, strides=1, padding='SAME', 
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn5 = tf.layers.batch_normalization(conv5, training=batch_norm, fused=True)
        relu5 = tf.nn.relu(bn5)        # 4x4x4096

        upsample1 = tf.image.resize_images(relu5, [7, 7])


        conv6 = tf.layers.conv2d(upsample1, 256, 3, strides=1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn6 = tf.layers.batch_normalization(conv6, training=batch_norm, fused=True)
        relu6 = tf.nn.relu(bn6)
        conv7 = tf.layers.conv2d(relu6, 256, 3, strides=1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn7 = tf.layers.batch_normalization(conv7, training=batch_norm, fused=True)
        relu7 = tf.nn.relu(bn7)

        upsample2 = tf.image.resize_images(relu7, [14, 14])


        conv8 = tf.layers.conv2d(upsample2, 256, 3, strides=1, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn8 = tf.layers.batch_normalization(conv8, training=batch_norm, fused=True)
        app_embed = tf.nn.relu(bn8)                       # 14x14x256

        return app_embed


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


def heatmap_features1(pose_maps, n_filters, reuse=tf.AUTO_REUSE):

    with tf.variable_scope('heatmap_features1', reuse=reuse):
        batch_norm = True

        conv1 = tf.layers.conv2d(pose_maps, n_filters, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn1 = tf.layers.batch_normalization(conv1, training=batch_norm, fused=True)
        relu1 = tf.nn.relu(bn1)
        conv2 = tf.layers.conv2d(relu1, n_filters, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn2 = tf.layers.batch_normalization(conv2, training=batch_norm, fused=True)
        relu2 = tf.nn.relu(bn2)

        return relu2


def heatmap_features2(pose_maps, n_filters, reuse=tf.AUTO_REUSE):

    with tf.variable_scope('heatmap_features2', reuse=reuse):
        batch_norm = True

        conv1 = tf.layers.conv2d(pose_maps, n_filters, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn1 = tf.layers.batch_normalization(conv1, training=batch_norm, fused=True)
        relu1 = tf.nn.relu(bn1)
        conv2 = tf.layers.conv2d(relu1, n_filters, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn2 = tf.layers.batch_normalization(conv2, training=batch_norm, fused=True)
        relu2 = tf.nn.relu(bn2)

        return relu2


def heatmap_features3(pose_maps, n_filters, reuse=tf.AUTO_REUSE):

    with tf.variable_scope('heatmap_features3', reuse=reuse):
        batch_norm = True

        conv1 = tf.layers.conv2d(pose_maps, n_filters, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn1 = tf.layers.batch_normalization(conv1, training=batch_norm, fused=True)
        relu1 = tf.nn.relu(bn1)
        conv2 = tf.layers.conv2d(relu1, n_filters, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn2 = tf.layers.batch_normalization(conv2, training=batch_norm, fused=True)
        relu2 = tf.nn.relu(bn2)

        return relu2

        
def heatmap_features4(pose_maps, n_filters, reuse=tf.AUTO_REUSE):

    with tf.variable_scope('heatmap_features4', reuse=reuse):
        batch_norm = True

        conv1 = tf.layers.conv2d(pose_maps, n_filters, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn1 = tf.layers.batch_normalization(conv1, training=batch_norm, fused=True)
        relu1 = tf.nn.relu(bn1)

        conv2 = tf.layers.conv2d(relu1, n_filters, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn2 = tf.layers.batch_normalization(conv2, training=batch_norm, fused=True)
        relu2 = tf.nn.relu(bn2)

        return relu2


def decoder_image(pose_app_embedding, pose_embedding, reuse=tf.AUTO_REUSE):

    with tf.variable_scope('unified_decoder', reuse=reuse):
        batch_norm = True
        
        with tf.variable_scope('decoder_init', reuse=reuse):
        
            # block1
            conv1 = tf.layers.conv2d(pose_app_embedding, 512, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn1 = tf.layers.batch_normalization(conv1, training=batch_norm, fused=True)
            relu1 = tf.nn.relu(bn1)
            conv2 = tf.layers.conv2d(relu1, 512, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn2 = tf.layers.batch_normalization(conv2, training=batch_norm, fused=True)
            relu2 = tf.nn.relu(bn2)

            upsample1 = tf.image.resize_images(relu2, [28, 28])
            

            # block2
            conv3 = tf.layers.conv2d(upsample1, 256, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn3 = tf.layers.batch_normalization(conv3, training=batch_norm, fused=True)
            relu3 = tf.nn.relu(bn3)
            conv4 = tf.layers.conv2d(relu3, 256, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn4 = tf.layers.batch_normalization(conv4, training=batch_norm, fused=True)
            relu4 = tf.nn.relu(bn4)

            upsample2 = tf.image.resize_images(relu4, [56, 56])
        

        with tf.variable_scope('decoder_I', reuse=reuse):
            block3_input = tf.concat([upsample2, pose_embedding[1]], 3)

            # block3
            conv5 = tf.layers.conv2d(block3_input, 128, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn5 = tf.layers.batch_normalization(conv5, training=batch_norm, fused=True)
            relu5 = tf.nn.relu(bn5)
            conv6 = tf.layers.conv2d(relu5, 128, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn6 = tf.layers.batch_normalization(conv6, training=batch_norm, fused=True)
            relu6 = tf.nn.relu(bn6)

            upsample3 = tf.image.resize_images(relu6, [112, 112])
            

            block4_input = tf.concat([upsample3, pose_embedding[0]], 3)

            #block4
            conv7 = tf.layers.conv2d(block4_input, 64, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn7 = tf.layers.batch_normalization(conv7, training=batch_norm, fused=True)
            relu7 = tf.nn.relu(bn7)
            conv8 = tf.layers.conv2d(relu7, 64, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn8 = tf.layers.batch_normalization(conv8, training=batch_norm, fused=True)
            relu8 = tf.nn.relu(bn8)

            upsample4 = tf.image.resize_images(relu8, [224, 224])


            # output image
            conv9 = tf.layers.conv2d(upsample4, 64, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn9 = tf.layers.batch_normalization(conv9, training=batch_norm, fused=True)
            relu9 = tf.nn.relu(bn9)
            pose_app_image = tf.layers.conv2d(relu9, 3, 3, strides=1, padding='SAME', \
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

        return pose_app_image, upsample1


def decoder_seg(seg_features, pose_embedding, reuse=tf.AUTO_REUSE):

    with tf.variable_scope('decoder_seg', reuse=reuse):
        batch_norm = True

        # block1
        conv1 = tf.layers.conv2d(seg_features, 256, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn1 = tf.layers.batch_normalization(conv1, training=batch_norm, fused=True)
        relu1 = tf.nn.relu(bn1)
        conv2 = tf.layers.conv2d(relu1, 256, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn2 = tf.layers.batch_normalization(conv2, training=batch_norm, fused=True)
        relu2 = tf.nn.relu(bn2)

        upsample1 = tf.image.resize_images(relu2, [56, 56])
        

        block2_input = tf.concat([upsample1, pose_embedding[3]], 3)

        # block2
        conv3 = tf.layers.conv2d(block2_input, 128, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn3 = tf.layers.batch_normalization(conv3, training=batch_norm, fused=True)
        relu3 = tf.nn.relu(bn3)
        conv4 = tf.layers.conv2d(relu3, 128, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn4 = tf.layers.batch_normalization(conv4, training=batch_norm, fused=True)
        relu4 = tf.nn.relu(bn4)

        upsample2 = tf.image.resize_images(relu4, [112, 112])


        # output image
        conv5 = tf.layers.conv2d(upsample2, 64, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn5 = tf.layers.batch_normalization(conv5, training=batch_norm, fused=True)
        relu5 = tf.nn.relu(bn5)
        seg_mask = tf.layers.conv2d(relu5, 13, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

        part_segments = tf.nn.softmax(seg_mask, 3)
        fg_segment = tf.nn.softmax(tf.stack([tf.reduce_max(seg_mask[:,:,:,:12], 3), seg_mask[:,:,:,12]], 3), 3)

        return seg_mask, part_segments, fg_segment
