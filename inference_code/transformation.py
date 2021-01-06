import tensorflow as tf
import math

from config import opts
from commons_17 import prior_sk_data as sk_data

def unit_norm_tf(mat, dim=1):
    norm = (tf.sqrt(tf.reduce_sum(mat ** 2, dim)) + 1e-9)
    norm = tf.expand_dims(norm, dim)
    mat = mat / norm
    return mat

def get_skeleton_transform_matrix(skeleton_batch):
   # skeleton_batch: [B, 15, 3]
   right_hip = sk_data.get_joint_index('right_hip')
   left_hip = sk_data.get_joint_index('left_hip')
   neck = sk_data.get_joint_index('neck')

   r = skeleton_batch[:, right_hip:right_hip + 1]
   l = skeleton_batch[:, left_hip:left_hip + 1]
   n = skeleton_batch[:, neck:neck + 1]

   m = 0.5 * (r + l)
   z_ = unit_norm_tf(n - m, dim=-1)
   y_ = unit_norm_tf(tf.cross(l - r, n - r), dim=-1)
   x_ = tf.cross(y_, z_)

   transform_mats = tf.concat([x_, y_, z_], axis=1)
   print ('dets', tf.linalg.det(transform_mats))
   print ('trasnforms_mats', transform_mats.shape)

   return transform_mats

def root_relative_to_view_norm(skeleton_batch, transform_mats):
   return tf.matmul(skeleton_batch, transform_mats, transpose_b=True)

def get_rotmat_camera(alpha, beta, gamma):
    zero = tf.constant([0.])
    one = tf.constant([1.])
    one_zero_zero = tf.concat([one, zero, zero], 0)
    zero_one_zero = tf.concat([zero, one, zero], 0)
    zero_zero_one = tf.concat([zero, zero, one], 0)
    # print
    R_x = tf.stack([one_zero_zero,
                       tf.stack([zero[0], tf.cos(alpha), -tf.sin(alpha)], 0, name='R_x1'),
                       tf.stack([zero[0], tf.sin(alpha), tf.cos(alpha)], 0, name='R_x2')], 0, name='R_x')

    #     print "Rx", R_x
    R_y = tf.stack([tf.stack([tf.cos(beta), zero[0], tf.sin(beta)], 0, name='R_y0'),
                       zero_one_zero,
                       tf.stack([-tf.sin(beta), zero[0], tf.cos(beta)], 0, name='R_y2')], 0, name='R_y')

    #     print "Ry", R_y
    # try with R_z as identity
    R_z = tf.stack([tf.stack([tf.cos(gamma), -tf.sin(gamma), zero[0]], 0, name='R_z0'),
                       tf.stack([tf.sin(gamma), tf.cos(gamma), zero[0]], 0, name='R_z1'),
                       zero_zero_one], 0, name='R_z')

    #     print "Rz", R_z
    R = tf.matmul(R_x, tf.matmul(R_y, R_z))
    return R


def get_rotmat_batch_camera(angles_batch):
    rotmats = []
    for xx in range(opts['batch_size']):
        alpha = angles_batch[xx][0]
        beta = angles_batch[xx][1]
        gamma = angles_batch[xx][2]
        rotmat = get_rotmat_camera(alpha, beta, gamma)
        rotmats.append(rotmat)
    return tf.stack(rotmats, 0, name='rotmat')


def get_projected_points_pers(P, R, T, out_cam_params): #, cam_center, offset):
    with tf.variable_scope('get_projected_points'):
        img_siz = float(opts['image_size'])
        # rotate
        # X = tf.matmul(R, tf.transpose(P,perm=[0,2,1]) - T)
        X = tf.matmul(R, tf.transpose(P,perm=[0,2,1]))
        X = X - T
        # translate
        # X = X - tf.to_float(T) # tf.transpose(c,perm=[0,2,1])
        
        unscaled_XX = tf.stack([X[:,0, :] / X[:,1, :], X[:,2, :] / X[:,1, :]], 1)

        max_x = tf.reduce_max(unscaled_XX[:,0:1,:], 2)
        max_y = tf.reduce_max(unscaled_XX[:,1:2,:], 2)
        min_x = tf.reduce_min(unscaled_XX[:,0:1,:], 2)
        min_y = tf.reduce_min(unscaled_XX[:,1:2,:], 2)
        dist_x = max_x - min_x
        dist_y = max_y - min_y
        max_dist = tf.maximum(dist_x, dist_y)
        print ('max_dist', max_dist)
        # f = tf.expand_dims((tf.nn.tanh(out_cam_params[:,0:1]) + 1.0) * img_siz / (2.0 * max_dist), 1)   # Bx1x1
        f = tf.expand_dims((img_siz-20.0) / max_dist, 1)   # Bx1x1
        # f = (img_siz-20.0) / tf.stack([dist_x, dist_y], 1)
        print ("f ", f)

        # max_x = tf.reduce_max(unscaled_XX[:,0:1,:], 2) - tf.reduce_min(unscaled_XX[:,0:1,:], 2)
        # max_y = tf.reduce_max(unscaled_XX[:,1:2,:], 2) - tf.reduce_min(unscaled_XX[:,1:2,:], 2)
        # XX = f * (unscaled_XX - unscaled_XX[:,:,0:1]) + c
        # scaling_factor = (f / tf.expand_dims(tf.concat([max_x, max_y],1),2))
        # XX = scaling_factor * unscaled_XX + c
        scaled_XX = f * (unscaled_XX - tf.stack([min_x ,min_y], 1))
        min_x = tf.reduce_min(scaled_XX[:,0:1,:], 2)
        min_y = tf.reduce_min(scaled_XX[:,1:2,:], 2)
        max_x = tf.reduce_max(scaled_XX[:,0:1,:], 2)
        max_y = tf.reduce_max(scaled_XX[:,1:2,:], 2)
        # cc = tf.expand_dims(tf.nn.sigmoid(out_cam_params[:,1:3]) * (img_siz - tf.concat([min_x, min_y], 1)), 2) # Bx2x1
        # cc = tf.expand_dims(- tf.concat([min_x, min_y], 1) + tf.nn.sigmoid(out_cam_params[:,0:2]) * (img_siz - tf.concat([max_x, max_y], 1)), 2) # Bx2x1
        cc = tf.expand_dims(tf.nn.sigmoid(out_cam_params[:,0:2]) * (img_siz - tf.concat([max_x, max_y], 1)), 2) # Bx2x1
        print ("cc ", cc)
        XX = scaled_XX + cc

    return tf.transpose(unscaled_XX, (0,2,1)), tf.transpose(XX, (0,2,1)), tf.transpose(X, (0,2,1)), f


def get_2d_skeleton(vars_dict):
    """
    skeleton_3d: Bx17x3
    out_cam_params: Bx3
    T: Bx3x1
    """
    skeleton_3d = vars_dict['pred_ske']
    out_cam_params = vars_dict['out_cam_params']
    out_cam_angles = vars_dict['out_cam_angles']
    T = vars_dict['trans']

    alpha = tf.reshape(out_cam_angles[:, 0], (-1, 1))
    beta = tf.reshape(out_cam_angles[:, 1], (-1, 1))/2.
    gamma = tf.reshape(out_cam_angles[:, 2], (-1, 1))
    
    print ("alpha, beta, gamma ", alpha, beta, gamma)
    angles_batch = tf.concat([alpha, beta, gamma], 1)
    print ("angles_batch ", angles_batch)
    R = get_rotmat_batch_camera(angles_batch)
    # T = tf.reshape(tf.constant([0.0, 0.0, 0.0]), (3,1))
    # f = tf.reshape(tf.constant([float(opts['image_size']), float(opts['image_size'])]), (2,1))
    # c = tf.reshape(tf.constant([float(opts['image_size'])/2., float(opts['image_size'])/2.]), (2,1))
    # f = tf.reshape(f, (-1, 2, 1))

    # T = tf.reshape(out_cam_params[:, 3:], (-1,3,1))
    print ("T ", T)
    print ("skeleton_3d ", skeleton_3d)
    # projs_2d, rot_ske = get_projected_points(skeleton_3d, R, T, f, c)    # output y,x i.e y values greatly varying and x values not so much
    unscaled_projs_2d, projs_2d, rot_ske, focal_length = get_projected_points_pers(skeleton_3d, R, T, out_cam_params)
    print ("projs_2d ", projs_2d)
    return unscaled_projs_2d, projs_2d, rot_ske, focal_length


def transform_3d_2d(vars_dict):
    
    unscaled_skeleton_2d, skeleton_2d, rot_ske, focal_length = get_2d_skeleton(vars_dict)
    # skeleton_2d_unscaled = tf.stack([skeleton_2d_unscaled[:,:,1], skeleton_2d_unscaled[:,:,0]], 2)
    ret_dict = {}
    ret_dict['rot_ske'] = rot_ske
    ret_dict['projs_skeleton_2d'] = skeleton_2d
    ret_dict['unscaled_projs_skeleton_2d'] = unscaled_skeleton_2d
    ret_dict['focal_length'] = focal_length
    return ret_dict


def make_skeleton(pred_3d, out_angle):

    pelvis = tf.ones_like(pred_3d[:,0,:])*tf.constant([[0.0, 0.0, 0.0]])
    print ("pelvis ", pelvis)
    rhip_init = tf.ones_like(pred_3d[:,0,:])*tf.constant([[1.05, 0.0, 0.0]])
    lhip_init = tf.ones_like(pred_3d[:,0,:])*tf.constant([[-1.05, 0.0, 0.0]])
    hip_angle = tf.tanh(out_angle)[:,0]
    print ("hip_angle ", hip_angle)
    theta = hip_angle * math.pi/4.
    zero = tf.zeros_like(hip_angle)
    one = tf.ones_like(hip_angle)
    hip_rot_mat = tf.stack([tf.stack([tf.cos(theta), zero, tf.sin(theta)], 1),
                                tf.stack([zero, one, zero], 1),
                                tf.stack([-tf.sin(theta), zero, tf.cos(theta)], 1)], 1)
    print ("hip_rot_mat ", hip_rot_mat)

    rhip = tf.matmul(hip_rot_mat, tf.expand_dims(rhip_init, 2))
    lhip = tf.matmul(hip_rot_mat, tf.expand_dims(lhip_init, 2))
    rhip = tf.reshape(rhip, (tf.shape(rhip)[0], rhip.shape[1]))
    lhip = tf.reshape(lhip, (tf.shape(lhip)[0], lhip.shape[1]))
    print ("lhip ", lhip)
    print ("rhip ", rhip)

    n = tf.ones_like(pred_3d[:,0,:])*tf.constant([[0.0, 0.0, 4.75]])

    rs = unit_norm_tf(pred_3d[:,0,:], dim=1)*1.37 + n
    re = unit_norm_tf(pred_3d[:,1,:], dim=1)*2.8 + rs
    rh = unit_norm_tf(pred_3d[:,2,:], dim=1)*2.4 + re
    ls = unit_norm_tf(pred_3d[:,3,:], dim=1)*1.37 + n
    le = unit_norm_tf(pred_3d[:,4,:], dim=1)*2.8 + ls
    lh = unit_norm_tf(pred_3d[:,5,:], dim=1)*2.4 + le
    head = unit_norm_tf(pred_3d[:,6,:], dim=1)*2.0 + n
    rk = unit_norm_tf(pred_3d[:,7,:], dim=1)*4.2 + rhip
    ra = unit_norm_tf(pred_3d[:,8,:], dim=1)*3.6 + rk
    rf = unit_norm_tf(pred_3d[:,9,:], dim=1)*2.0 + ra
    lk = unit_norm_tf(pred_3d[:,10,:], dim=1)*4.2 + lhip
    la = unit_norm_tf(pred_3d[:,11,:], dim=1)*3.6 + lk
    lf = unit_norm_tf(pred_3d[:,12,:], dim=1)*2.0 + la

    
    skeleton = tf.concat([tf.expand_dims(pelvis, axis=1), tf.expand_dims(n, axis=1), tf.expand_dims(rs, axis=1), tf.expand_dims(re, axis=1),\
                          tf.expand_dims(rh, axis=1), tf.expand_dims(ls, axis=1), tf.expand_dims(le, axis=1), tf.expand_dims(lh, axis=1),\
                          tf.expand_dims(head, axis=1), tf.expand_dims(rhip, axis=1), tf.expand_dims(rk, axis=1), tf.expand_dims(ra, axis=1), \
                          tf.expand_dims(rf, axis=1), tf.expand_dims(lhip, axis=1), tf.expand_dims(lk, axis=1), tf.expand_dims(la, axis=1), \
                          tf.expand_dims(lf, axis=1)], axis=1, name='skeleton')

    return skeleton
