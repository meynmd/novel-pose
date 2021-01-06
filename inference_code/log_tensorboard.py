import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw
from io import BytesIO
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io

from config import opts

def concatenate_images(images, dim):
    all_images = images[0]
    #print ("Reaching here")
    for i in range(1,len(images)):
      all_images = np.concatenate([all_images, images[i]], dim)
    #print ("Not reaching here")
    return all_images

def skeleton_image(joints_2d, img):
    img_copy = img.copy()
    colors = [(0,0,255), (0,255,0), (255,0,0), (255,0,255), (0,255,255), (255,255,0), (127,127,0), (0,127,0), (100,0,100), 
          (255,0,255), (0,255,0), (0,0,255), (255,255,0), (127,127,0), (100,0,100), (175,100,195), (255,125,25)]
    # joints_2d = joints_2d * opts['image_size']/2.0 + opts['image_size']/2.0
    for i in range(joints_2d.shape[0]):
        #         ax.text(joints_2d[i, 0], -joints_2d[i, 1], str(i))
        x_pair = [joints_2d[i, 0], joints_2d[limb_parents[i], 0]]
        y_pair = [joints_2d[i, 1], joints_2d[limb_parents[i], 1]]
        img_copy = cv2.line(img_copy, (int(x_pair[0]),int(y_pair[0])), (int(x_pair[1]),int(y_pair[1])), colors[i],4)

    return img_copy

def log_images(tag, image, step, writer):
    """Logs a list of images."""

    height, width, channel = image.shape
    image = Image.fromarray(image)
    output = BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    # Create an Image object
    img_sum = tf.Summary.Image(height=height,
                               width=width,
                               colorspace=channel,
                               encoded_image_string=image_string)
    # Create a Summary value
    im_summary = tf.Summary.Value(tag='%s' % (tag), image=img_sum)

    # Create and write Summary
    summary = tf.Summary(value=[im_summary])
    writer.add_summary(summary, step)

def get_ax(joints_3d, fig, az=0, ele=10, subplot='111'):
    ax = fig.add_subplot(subplot, projection='3d')

    lim_max_x = np.amax(joints_3d[:, 0])
    lim_min_x = np.amin(joints_3d[:, 0])
    lim_max_y = np.amax(joints_3d[:, 1])
    lim_min_y = np.amin(joints_3d[:, 1])
    lim_max_z = np.amax(joints_3d[:, 2])
    lim_min_z = np.amin(joints_3d[:, 2])
    # print (lim_max_x, lim_min_x)
#     print("lim", lim)
    ax.view_init(azim=az, elev=ele)
    
    ax.set_xlim(lim_min_x-1.0, lim_max_x+1.0)
    ax.set_ylim(lim_min_y-1.0, lim_max_y+1.0)
    ax.set_zlim(lim_min_z-1.0, lim_max_z+1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    return ax

limb_parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 11, 0, 13, 14, 15]

def draw_limbs_3d_plt(joints_3d, ax, limb_parents=limb_parents, z_flip = True):
    for i in range(joints_3d.shape[0]):
        ax.text(joints_3d[i, 0], joints_3d[i, 1], joints_3d[i, 2], s=str(i))
        x_pair = [joints_3d[i, 0], joints_3d[limb_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[limb_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[limb_parents[i], 2]]
        if z_flip:
            ax.plot(z_pair, x_pair, y_pair, linewidth=3, antialiased=True)
        else:
            ax.plot(x_pair, y_pair,z_pair, linewidth=3, antialiased=True)
        dist = np.sqrt(np.square(x_pair[0]-x_pair[1]) + np.square(y_pair[0]-y_pair[1]) + np.square(z_pair[0]-z_pair[1]))

def get_skeleton_plot(joints_3d, ax, limb_parents=limb_parents, title="", z_flip=True):
    draw_limbs_3d_plt(joints_3d, ax, limb_parents, z_flip=False)
    plt.title(title)

def plot_skeleton(joints_3d, ax, limb_parents=limb_parents, title="", z_flip=True):
    get_skeleton_plot(joints_3d, ax, limb_parents, title, z_flip=z_flip)

def vis_3d(skeleton_3d, title):
    fig = plt.figure(frameon=False, figsize=(10, 10))

    ax = get_ax(skeleton_3d, fig, az=90, subplot='121')
    plot_skeleton(skeleton_3d, ax, z_flip=False)
    ax.set_title(title+"_front_view")

    ax = get_ax(skeleton_3d, fig, az=60, ele=20, subplot='122')
    plot_skeleton(skeleton_3d, ax, z_flip=False)
    ax.set_title(title+"_side_view")

    return fig

def skeleton_to_image(s, title):
    fig = vis_3d(s[0], title)
    fig.savefig("skeleton_3d.png")
    plt.close(fig)
    fig_img = cv2.imread("skeleton_3d.png")[:, :, ::-1]

    return fig_img

def get_3d_skeleton_images(skeletons, titles):

    for i, sk in enumerate(skeletons):
      if i==0:
        skeleton_images = skeleton_to_image(sk, titles[i])
      else:
        skeleton_images = np.concatenate([skeleton_images, skeleton_to_image(sk, titles[i])], 1)
    return skeleton_images

def log_tensorboard(return_dict):

        source_images = return_dict['src_im'][0]

        target_images = return_dict['tgt_im'][0] + (np.array([103.939, 116.779, 123.68])/255.)

        recons_image = return_dict['img_pt_as'][0]

        tgt_sk_image = skeleton_image(return_dict['tgt_projs_skeleton_2d'][0], np.zeros((opts['image_size'],opts['image_size'],3))*255.)

        tgt_sk_overlaid_recons = skeleton_image(return_dict['tgt_projs_skeleton_2d'][0], recons_image*255.)

        fig_img1 = concatenate_images([ source_images*255., target_images*255.,
                                    recons_image*255.,  tgt_sk_overlaid_recons], 1)
        
        cv2.imwrite("predictions.jpg", fig_img1[:,:,::-1].astype(np.uint8))
	print ("results logged")

        titles = ["pred_3d", "rot_3d"]
        syn2_skeletons = get_3d_skeleton_images([return_dict['tgt_pred_ske'], return_dict['tgt_rot_ske']], titles)
	cv2.imwrite("pred_skeletons.jpg", syn2_skeletons.astype(np.uint8))
        print ("gt_skeletons logged")
