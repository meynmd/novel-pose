# ==========================================================
# Author: Siddharth Seth
# ==========================================================
from __future__ import print_function
from __future__ import absolute_import
import os,sys
import json
import time
import tensorflow as tf
import os.path as osp
import os
import cv2
import numpy as np
import random
import copy
import scipy.io as sio
import imutils

import cnn_train as tru
from config import opts
from data_loader import get_feed_dict


def main(args):
  # get the data and logging (checkpointing) directories:
  log_dir = 'log_dir'
  NUM_STEPS = 3000000


  graph = tf.Graph()
  with graph.as_default():
    batch_size = opts['batch_size']
    src_im = tf.placeholder(tf.float32, shape = [None, 224,224,3])    
    future_im = tf.placeholder(tf.float32, shape = [None, 224,224,3])
    pose_src_centered =  tf.placeholder(tf.float32, shape = [None, 12, 4])
    pose_src =  tf.placeholder(tf.float32, shape = [None, 12, 4])
    parts_src_centered = tf.placeholder(tf.float32, [None, 12, 112, 112, 3])
    unc_parts_src_centered = tf.placeholder(tf.float32, [None, 12, 112, 112, 3])

    inputs = {'source_im': src_im, 'future_im': future_im, 'pose_src_centered': pose_src_centered, 'parts_src_centered': parts_src_centered, 'pose_src': pose_src,'unc_parts_src_centered': unc_parts_src_centered}
    feed_dict = get_feed_dict(args, inputs)

    # create the network distributed over multi-GPUs:
    ret_dict = tru.train_single(graph, inputs)

    tru.train_loop(graph, ret_dict, feed_dict)

if  __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Run Inference on the given images')
  parser.add_argument("src_im", metavar = "srcim", type = str, help = "the path to image with source appearance")
  parser.add_argument("tgt_im", metavar = "tgtim", type = str, help = "the path to image with the target pose")
  args = parser.parse_args()
  main(args)
