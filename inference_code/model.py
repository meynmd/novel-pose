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

from networks import app_encoder, pose_encoder
from transformation import transform_3d_2d, get_skeleton_transform_matrix, root_relative_to_view_norm, make_skeleton
from utils import get_reconstructed_image, gaussian_heatmaps, colorize_landmark_maps, limb_maps
from config import opts
from stn import spatial_transformer_network as transformer


def depth_normalize(tgt_pred_ske):
    
    limb_points_3d = tf.concat([tf.concat([tgt_pred_ske[:,0:1], tgt_pred_ske[:,1:2]], 2),
                              tf.concat([tgt_pred_ske[:,9:10], tgt_pred_ske[:,10:11]], 2),
                              tf.concat([tgt_pred_ske[:,10:11], tgt_pred_ske[:,11:12]], 2),
                              tf.concat([tgt_pred_ske[:,11:12], tgt_pred_ske[:,12:13]], 2),
                              tf.concat([tgt_pred_ske[:,13:14], tgt_pred_ske[:,14:15]], 2),
                              tf.concat([tgt_pred_ske[:,14:15], tgt_pred_ske[:,15:16]], 2),
                              tf.concat([tgt_pred_ske[:,15:16], tgt_pred_ske[:,16:17]], 2),
                              tf.concat([tgt_pred_ske[:,1:2], tgt_pred_ske[:,8:9]], 2),
                              tf.concat([tgt_pred_ske[:,2:3], tgt_pred_ske[:,3:4]], 2),
                              tf.concat([tgt_pred_ske[:,3:4], tgt_pred_ske[:,4:5]], 2),
                              tf.concat([tgt_pred_ske[:,5:6], tgt_pred_ske[:,6:7]], 2),
                              tf.concat([tgt_pred_ske[:,6:7], tgt_pred_ske[:,7:8]], 2)], 1)
    limb_depths_x = (limb_points_3d[:,:,0] + limb_points_3d[:,:,3]) / 2.
    limb_depths_y = (limb_points_3d[:,:,1] + limb_points_3d[:,:,4]) / 2.
    limb_depths_z = (limb_points_3d[:,:,2] + limb_points_3d[:,:,5]) / 2.
    
    return limb_points_3d, limb_depths_y
    
    
def normalize_depth_maps(phi_p, limb_depths_y):
    dl = limb_depths_y
    dl = ((dl - tf.reduce_min(dl)) / (tf.reduce_max(dl) - tf.reduce_min(dl)) ) + 0.001
    
    tf_phi_p = tf.divide(phi_p, tf.expand_dims(tf.expand_dims(dl, 1), 1))

    phi_d_l = tf.nn.softmax(tf_phi_p, axis=3)
    
    phi_d_bg = 1 - tf.reduce_max(phi_d_l, axis=3, keepdims=True)

    phi_d = tf.concat([phi_d_bg, phi_d_l],axis=3)			#### bg first to handle p_chl=2 case when bg and softmax low value is same
    final_phi_d = tf.nn.softmax(phi_d, axis=3)
    
    return final_phi_d
    

def get_ang(temp_lineA, temp_lineB):

    lineA = temp_lineA#.astpye(np.float32)
    lineB = temp_lineB#.astpye(np.float32)

    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]


    t1 = tf.atan2(vA[1], vA[0]+1e-7)
    t2 = tf.atan2(vB[1], vB[0]+1e-7)


    #if t1 < 0.:
    t1 = 2. * np.pi + t1   #if the angles are negative, make them positive
    #if t2 < 0.:
    t2 = 2. * np.pi + t2

    return (t1-t2) #* 180. / np.pi

    
def get_dist(x1, y1, x2, y2):
    return tf.sqrt(tf.pow(x2-x1, 2)+tf.pow(y2-y1, 2))

def stn_block(name, theta, inp):
    with tf.variable_scope(name):
        theta = tf.reshape(theta, (-1, 2*3))

        # define loc net weight and bias
        loc_in = 112*112*3
        loc_out = 6
        W_loc = tf.Variable(tf.zeros([loc_in, loc_out]), name='W_loc')
        b_loc = theta

        # tie everything together
        fc_loc = tf.matmul(tf.zeros([opts['batch_size']*12, loc_in]), W_loc) + b_loc            # [B*12, 6]
        op = transformer(inp, fc_loc)

    return op

def get_transformed_parts(pose_2d_rnd, maps, pose_src_centered, name):
    """
        transforms the maps B*12x112x112x3, based on the affine transformations obtained from pose_src_centered and pose_2d_rnd
        where pose_2d_rnd is the final pose and pose_src_centered is the initial pose

    """
    with tf.variable_scope(name):

        limb_points_rnd = tf.concat([tf.concat([pose_2d_rnd[:,0:1], pose_2d_rnd[:,1:2]], 2),
                                tf.concat([pose_2d_rnd[:,9:10], pose_2d_rnd[:,10:11]], 2),
                                tf.concat([pose_2d_rnd[:,10:11], pose_2d_rnd[:,11:12]], 2),
                                tf.concat([pose_2d_rnd[:,11:12], pose_2d_rnd[:,12:13]], 2),
                                tf.concat([pose_2d_rnd[:,13:14], pose_2d_rnd[:,14:15]], 2),
                                tf.concat([pose_2d_rnd[:,14:15], pose_2d_rnd[:,15:16]], 2),
                                tf.concat([pose_2d_rnd[:,15:16], pose_2d_rnd[:,16:17]], 2),
                                tf.concat([pose_2d_rnd[:,1:2], pose_2d_rnd[:,8:9]], 2),
                                tf.concat([pose_2d_rnd[:,2:3], pose_2d_rnd[:,3:4]], 2),
                                tf.concat([pose_2d_rnd[:,3:4], pose_2d_rnd[:,4:5]], 2),
                                tf.concat([pose_2d_rnd[:,5:6], pose_2d_rnd[:,6:7]], 2),
                                tf.concat([pose_2d_rnd[:,6:7], pose_2d_rnd[:,7:8]], 2)], 1)       # Bx12x4

        mid_points_x = (limb_points_rnd[:,:,0] + limb_points_rnd[:,:,2]) // 2.
        mid_points_y = (limb_points_rnd[:,:,1] + limb_points_rnd[:,:,3]) // 2.

        limb_points_rnd_centered = tf.stack([limb_points_rnd[:,:,0]-mid_points_x, limb_points_rnd[:,:,1]-mid_points_y,
                                            limb_points_rnd[:,:,2]-mid_points_x, limb_points_rnd[:,:,3]-mid_points_y], 2)   + 56    # Bx12x4 , center the points at (56, 56) (center of 112, 112)

        x1, y1 = pose_src_centered[:,:,0], pose_src_centered[:,:,1]
        x1_, y1_ = limb_points_rnd_centered[:,:,0], limb_points_rnd_centered[:,:,1]
        angle = get_ang([[x1, y1], [tf.ones_like(x1)*56., tf.ones_like(x1)*56.]],
                        [[x1_, y1_], [tf.ones_like(x1)*56., tf.ones_like(x1)*56.]])     # Bx12

        q =  get_dist(pose_src_centered[:,:,0], pose_src_centered[:,:,1], pose_src_centered[:,:,2], pose_src_centered[:,:,3]) / ( get_dist(limb_points_rnd[:,:,0], limb_points_rnd[:,:,1], limb_points_rnd[:,:,2], limb_points_rnd[:,:,3])+ 1e-7)
        
        p = tf.ones(1)
        a11 = tf.cos(angle)
        a12 = tf.sin(angle)
        a21 = tf.sin(angle)
        a22 = tf.cos(angle)

        #scale only
        theta = tf.stack([tf.stack([q, tf.zeros_like(a11), tf.zeros_like(a11)], 2), 
                    tf.stack([tf.zeros_like(a11), tf.ones_like(a11), tf.zeros_like(a11)],2)],2)
        h_trans_scale  = stn_block("scale_trans", theta, maps)
        
            
        # rotation only
        theta = tf.stack([tf.stack([a11, -a12, tf.zeros_like(a11)], 2), 
                    tf.stack([a21, a22, tf.zeros_like(a11)],2)],2)        # (16,12,2,3)   

        h_trans_rot = stn_block("rot_trans", theta, h_trans_scale)

        # translation only
        theta = tf.stack([tf.stack([tf.ones_like(a11), tf.zeros_like(a11), (56.-mid_points_x)/56.], 2), 
                            tf.stack([tf.zeros_like(a11), tf.ones_like(a11), (56.-mid_points_y)/56.],2)],2)
        
        h_trans = stn_block("translate_trans", theta, h_trans_rot)

    return tf.transpose(tf.reshape(tf.reduce_max(h_trans, 3), (-1, 12, 112, 112)), perm=[0,2,3,1])


def model(inputs, costs_collection='costs'):

    # process input variables
        ret_dict = {}
        img_siz = opts['image_size']
        n_joints = opts['n_joints']

        src_im = inputs['source_im']
        tgt_im = inputs['future_im']
        pose_src_centered = inputs['pose_src_centered']
        parts_src_centered = inputs['parts_src_centered']
        pose_src = inputs['pose_src']
        unc_parts_src_centered = inputs['unc_parts_src_centered']
        

	#add all the inputs to the return dict
        ret_dict['src_im'] = tf.reshape(src_im, (tf.shape(src_im)[0],img_siz,img_siz,3))
        ret_dict['tgt_im'] = tf.reshape(tgt_im, (tf.shape(tgt_im)[0],img_siz,img_siz,3))
        ret_dict['pose_src_centered'] = tf.reshape(pose_src_centered, (tf.shape(pose_src_centered)[0], 12, 4))
        ret_dict['parts_src_centered'] = tf.reshape(parts_src_centered, (tf.shape(parts_src_centered)[0]*12, 112, 112, 3)) # B, 12, 112, 112, 3
        ret_dict['pose_src'] = tf.reshape(pose_src, (tf.shape(pose_src)[0], 12, 4))
        ret_dict['unc_parts_src_centered'] = tf.reshape(unc_parts_src_centered, (tf.shape(unc_parts_src_centered)[0]*12, 112, 112, 3)) # B, 12, 112, 112, 3

	#choose the gpu based on the number of gpus defined in the config file
        with slim.arg_scope(resnet_v1.resnet_arg_scope()), tf.device("/gpu:1" if len(opts['gpu_ids'])>1 else "/gpu:0"):
            _, inception_dict_tgt = resnet_v1.resnet_v1_50(ret_dict['tgt_im'], is_training=False)

	#subtract the imagenet mean for the source image before passing it to the resnet
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            _, inception_dict_src = resnet_v1.resnet_v1_50(ret_dict['src_im']-tf.constant([103.939, 116.779, 123.68][::-1])/255., is_training=False, reuse=True)

        with tf.device("/gpu:1" if len(opts['gpu_ids'])>1 else "/gpu:0"):
         src_app = app_encoder(inception_dict_src['resnet_v1_50/block2/unit_3/bottleneck_v1'], reuse=tf.AUTO_REUSE)  # 14x14x1024

         tgt_app = app_encoder(inception_dict_tgt['resnet_v1_50/block2/unit_3/bottleneck_v1'], reuse=tf.AUTO_REUSE)  # 14x14x1024 

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

        

        #divide the predicted pose by 2 to scale it to (112x112) image size
        h_trans_transpose = get_transformed_parts(out_dict["projs_skeleton_2d"]/2, ret_dict["parts_src_centered"],ret_dict["pose_src_centered"], "h_trans")
        h_trans_unc_transpose = get_transformed_parts( out_dict["projs_skeleton_2d"]/2, ret_dict["unc_parts_src_centered"],ret_dict["pose_src_centered"], "h_trans")

        ret_dict["h_transformed_parts"] = h_trans_transpose
        ret_dict["h_transformed_parts_unc"] =  h_trans_unc_transpose 
        
        phi_p = h_trans_transpose       # B x 112, 112, 12
        psi_p = tf.reduce_max(h_trans_unc_transpose, 3, keepdims=False)   # B x 112, 112, 12
        

        # depth normalize
        limb_points_3d, limb_depths_y = depth_normalize(ret_dict['tgt_pred_ske'])

        final_phi_d = normalize_depth_maps(phi_p, limb_depths_y)
        
        h_trans_seg_112 = final_phi_d
        h_trans_seg_56 = tf.image.resize_images(final_phi_d, [56, 56])
        h_trans_seg_14 = tf.image.resize_images(final_phi_d, [14, 14])

        with tf.device("/gpu:0"):
         img_pt_as= get_reconstructed_image([h_trans_seg_112, h_trans_seg_56, h_trans_seg_14], 
                                                              src_app, reuse=tf.AUTO_REUSE)

        ret_dict['beta'] = tf.constant(10.)
        ret_dict['img_pt_as'] = tf.clip_by_value(img_pt_as, 0., 1.)
        
    ############ calculate losses
        return ret_dict
