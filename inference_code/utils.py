import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
from networks import decoder_image, decoder_seg, heatmap_features1, heatmap_features2, heatmap_features3, heatmap_features4
from config import opts
import random
import numpy as np

def get_reconstructed_image(tgt_pose_maps, app, reuse=False):

    pose_embedding = []
    # 112, 56, 14
    n_filters = [32, 256, 64]
    # for nf in range(len(n_filters)):
    pose_embedding.append(heatmap_features1(tgt_pose_maps[0], n_filters[0]))    # 112, 112, 32
    pose_embedding.append(heatmap_features2(tgt_pose_maps[1], n_filters[1]))    # 56, 56, 256
    pose_embedding.append(heatmap_features3(tgt_pose_maps[2], n_filters[1]))    # 14, 14, 256
    pose_embedding.append(heatmap_features4(tgt_pose_maps[1], n_filters[2]))    # 56, 56, 64

    # concatenate src pose and app embedding
    pose_app_embedding = tf.concat([pose_embedding[2], app], 3)             # Bx16x16x(256+14)
    recons_image, upsample1 = decoder_image(pose_app_embedding, pose_embedding, reuse=reuse)    # Bx224x224x3
    # print ("tgt_pose_src_app_embedding ", tgt_pose_src_app_embedding)


    return recons_image

def _exp_running_avg(x, init_val=0.0, rho=0.99, name='x'):
    x_avg = variables.model_variable(name+'_agg', shape=x.shape, dtype=x.dtype,
                                     initializer=tf.constant_initializer(init_val, x.dtype),
                                     trainable=False,device='/cpu:0')
    w_update = 1.0 - rho
    x_new = x_avg + w_update * (x - x_avg)
    # update_op = tf.cond(training_pl, lambda: tf.assign(x_avg, x_new), lambda: tf.constant(0.0))
    update_op = tf.assign(x_avg, x_new)
    with tf.control_dependencies([update_op]):
        return tf.identity(x_new)


def get_gaussian_maps(mu, shape_hw, inv_std, mode='ankush'):
  """
  Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
  given the gaussian centers: MU [B, NMAPS, 2] tensor.

  STD: is the fixed standard dev.
  """
  with tf.name_scope(None, 'gauss_map', [mu]):
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))

    x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))

  if mode in ['rot', 'flat']:
    mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)

    y = tf.reshape(y, [1, 1, shape_hw[0], 1])
    x = tf.reshape(x, [1, 1, 1, shape_hw[1]])

    g_y = tf.square(y - mu_y)
    g_x = tf.square(x - mu_x)
    dist = (g_y + g_x) * inv_std**2
    dist1 = (g_y + g_x) * (inv_std*4)**2

    if mode == 'rot':
      g_yx = tf.exp(-dist)
      g_yx1 = tf.exp(-dist1)
    else:
      g_yx = tf.exp(-tf.pow(dist + 1e-5, 0.25))

  elif mode == 'ankush':
    y = tf.reshape(y, [1, 1, shape_hw[0]])
    x = tf.reshape(x, [1, 1, shape_hw[1]])

    g_y = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_y - y) * inv_std)))
    g_x = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_x - x) * inv_std)))

    g_y = tf.expand_dims(g_y, axis=3)
    g_x = tf.expand_dims(g_x, axis=2)
    g_yx = tf.matmul(g_y, g_x)  # [B, NMAPS, H, W]

  else:
    raise ValueError('Unknown mode: ' + str(mode))

  g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
  g_yx1 = tf.transpose(g_yx1, perm=[0, 2, 3, 1])
  return g_yx, g_yx1

# get gaussian hetmaps correspnding to 2^n dimensions
def gaussian_heatmaps(points):
    """
    points: xy format, [-1,1] range expected
    """
    gauss_std = 0.1
    gauss_xy = []
    gauss_xy1 = []
    map_sizes = [opts['image_size']//16, opts['image_size']//4, opts['image_size']//2, opts['image_size']]
    # points_exchanged = tf.concat([points[:,:,1:2], points[:,:,0:1]], 2, name='points_exchanged')
    points_exchanged = tf.stack([points[:,:,1], points[:,:,0]], 2, name='points_exchanged')/112. - 1.
    for map_size in map_sizes:
        gauss_xy_, gauss_xy_1 = get_gaussian_maps(points_exchanged, [map_size, map_size],
                                1.0 / gauss_std,                            # gauss_std = 0.1
                                mode='rot')                                 # mode = 'rot'
        gauss_xy.append(gauss_xy_)
        gauss_xy1.append(gauss_xy_1)
    

    return gauss_xy, gauss_xy1

def get_rot_gaussian_maps(mu, shape_hw, inv_std1, inv_std2, angles, mode='ankush'):
    """
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.

    STD: is the fixed standard dev.
    """
    with tf.name_scope(None, 'gauss_map', [mu]):
        mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]     # (B, 17, 1)

        y = tf.to_float(np.linspace(-1.0, 1.0, shape_hw[0]))

        x = tf.to_float(np.linspace(-1.0, 1.0, shape_hw[1]))  # Bx14
            
        y = tf.reshape(tf.tile(y, [shape_hw[0]]), (-1, shape_hw[0], shape_hw[0]))
        y = tf.expand_dims(y,0) * tf.ones((mu.shape[1], shape_hw[0], shape_hw[0]))
        
        x = tf.reshape(tf.tile(x, [shape_hw[1]]), (-1, shape_hw[1], shape_hw[1]))  # Bx14x14
        x = tf.expand_dims(x,0) * tf.ones((mu.shape[1], shape_hw[1], shape_hw[1]))  # Bx16x14x14
        x = tf.transpose(x, perm=[0,1,3,2])
        mu_y, mu_x = tf.expand_dims(mu_y, 3), tf.expand_dims(mu_x, 3) # Bx16x1x1
        
        y = y - mu_y
        x = x - mu_x  # Bx16x14x14
        
        if mode in ['rot', 'flat']:
        
            # apply rotation to the grid
            yx_stacked = tf.stack([tf.reshape(y,(-1, y.shape[1],y.shape[2]*y.shape[3])), 
                                   tf.reshape(x,(-1, x.shape[1],x.shape[2]*x.shape[3]))], 2)   # (B, 16, 2, 196)
            rot_mat = tf.stack([ tf.stack([tf.cos(angles), tf.sin(angles)],2), 
                                 tf.stack([-tf.sin(angles), tf.cos(angles)],2) ], 3)   # (B, 16, 2, 2)

            rotated = tf.matmul(rot_mat, yx_stacked)   # (B, 16, 2, 196)
            
            y_rot = rotated[:,:,0,:]   # (B, 16, 196)
            x_rot = rotated[:,:,1,:]   # (B, 16, 196)
                    
            y_rot = tf.reshape(y_rot, (-1, mu.shape[1],shape_hw[0],shape_hw[0]))   # (B, 16, 14, 14)
            x_rot = tf.reshape(x_rot, (-1, mu.shape[1],shape_hw[1],shape_hw[1]))   # (B, 16, 14, 14)


            g_y = tf.square(y_rot)   # (B, 16, 14, 14)
            g_x = tf.square(x_rot)   # (B, 16, 14, 14)

            inv_std1 = tf.expand_dims(tf.expand_dims(inv_std1, 2), 2) # Bx16x1x1
            inv_std2 = tf.expand_dims(tf.expand_dims(inv_std2, 2), 2) # Bx16x1x1
            dist = (g_y * inv_std1**2 + g_x * tf.to_float(inv_std2)**2)

            if mode == 'rot':
                g_yx = tf.exp(-dist)

            else:
                g_yx = tf.exp(-tf.pow(dist + 1e-5, 0.25))

        else:
            raise ValueError('Unknown mode: ' + str(mode))

        g_yx = tf.transpose(g_yx, perm=[0, 3, 2, 1])

        return g_yx

def get_limb_centers(joints_2d):

    limb_parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 11, 0, 13, 14, 15]
    # limbs = []
    angles_x = []
    angles_y = []
    limbs_x = []
    limbs_y = []
    limb_length = []
    for i in range(1, joints_2d.shape[1]):
        #         ax.text(joints_2d[i, 0], -joints_2d[i, 1], str(i))
        # print ("joints_2d ", joints_2d)
        x_pair = [joints_2d[:, i, 0], joints_2d[:, limb_parents[i], 0]]
        # print ("x_pair ", x_pair)
        y_pair = [joints_2d[:, i, 1], joints_2d[:, limb_parents[i], 1]]
        # print ("y_pair ", y_pair)
        limbs_x.append((x_pair[0]+x_pair[1])/2.)
        limbs_y.append((y_pair[0]+y_pair[1])/2.)
        limb_length.append(tf.sqrt((x_pair[0]-x_pair[1])**2 + (y_pair[0]-y_pair[1])**2))
        # print ("limb_length ", limb_length)
        # calculate slope, m = tan(theta)
        angles_x.append(x_pair[0]-x_pair[1]) # because y is represented as x
        angles_y.append(y_pair[0]-y_pair[1])
        # print ("angles ", angles)
    angles_x = tf.stack(angles_x, 1)
    angles_y = tf.stack(angles_y, 1)

    angles = tf.atan2(angles_x, angles_y+1e-7)       # x/y as pose is passed as (y,x)
    
    limbs_x = tf.stack(limbs_x, 1)
    limbs_y = tf.stack(limbs_y, 1)
    limbs = tf.stack([limbs_x, limbs_y], 2)
    limb_length = tf.stack(limb_length, 1)
    return limbs, angles, limb_length

def limb_maps(pose_points):
    points_exchanged = tf.stack([pose_points[:,:,1], pose_points[:,:,0]], 2, name='points_exchanged')/112. - 1.
    
    limb_centers_yx, angles, limb_length = get_limb_centers(points_exchanged)
    # decreasing the value of ratio increases the length of the gaussian
    length_ratios = tf.ones(opts['n_joints']-1)*2.
    # decreasing the value of ratio increases the width of the gaussian
    width_ratios = tf.constant([8., 25., 20., 25., 25., 20., 25., 12.,
                              20., 15., 20., 20., 20., 15., 20., 20.])*tf.ones_like(limb_length)

    map_sizes = [opts['image_size']//16, opts['image_size']//4, opts['image_size']//2, opts['image_size']]
    gauss_xy = []
    for map_size in map_sizes:
        rot_gauss_map = get_rot_gaussian_maps(limb_centers_yx, [map_size,map_size], width_ratios, length_ratios / limb_length, angles, mode='rot')
        gauss_xy.append(rot_gauss_map)
    return gauss_xy


def get_random_color(pastel_factor = 0.5):
  return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
  return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
  max_distance = None
  best_color = None
  for i in range(0,100):
    color = get_random_color(pastel_factor = pastel_factor)
    if not existing_colors:
      return color
    best_distance = min([color_distance(color,c) for c in existing_colors])
    if not max_distance or best_distance > max_distance:
      max_distance = best_distance
      best_color = color
  return best_color

def get_n_colors(n, pastel_factor=0.9):
  colors = []
  for i in range(n):
    colors.append(generate_new_color(colors,pastel_factor = 0.9))
  return colors

def colorize_landmark_maps(maps):
  """
  Given BxHxWxN maps of landmarks, returns an aggregated landmark map
  in which each landmark is colored randomly. BxHxWxN
  """
  n_maps = maps.shape.as_list()[-1]
  # get n colors:
  colors = get_n_colors(n_maps, pastel_factor=0.0)
  hmaps = [tf.expand_dims(maps[..., i], axis=3) * np.reshape(colors[i], [1, 1, 1, 3])
           for i in range(n_maps)]
  return tf.reduce_max(hmaps, axis=0)
