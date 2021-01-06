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
from data_loader import get_test_set, read_data


def main(args):

    # get the data and logging (checkpointing) directories:
    log_dir = 'log_dir'


    graph = tf.Graph()
    with graph.as_default():

        batch_size = opts['batch_size']

        dummy_test_data = get_test_set()

        # create the TFDS for each dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((dummy_test_data))

        num_parallel_calls = 32

        # map the datasets to their read functions
        test_dataset = test_dataset.map(lambda z: tf.py_func(read_data, [z], [tf.float32]*2), num_parallel_calls=num_parallel_calls)

        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.prefetch(2)

        # set up inputs
        handle_pl = tf.placeholder(tf.string, shape=[])

        # base iterator that will be used to feed data from all the datasets
        base_iterator = tf.data.Iterator.from_string_handle(handle_pl, test_dataset.output_types,test_dataset.output_shapes)
        
        future_im, pose_3d = base_iterator.get_next()
        inputs = {'future_im': future_im,  'pose_3d': pose_3d}
     
        # create the network distributed over multi-GPUs:
        ret_dict = tru.train_single(graph, inputs)

        tru.train_loop(graph,  ret_dict, inputs, handle_pl, test_dataset)

if  __name__=='__main__':
    main(None)
