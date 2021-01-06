from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import os
from log_tensorboard import log_tensorboard
from model import model
from config import opts

def tower_loss(inputs):
  """
  Args:
    inputs: An input dictionary with placeholders

  Returns:
     A return dictionary with output tensors
  """

  ret_dict = model(inputs, costs_collection='costs')
    
  return  ret_dict


def get_network_params(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def train_single(graph, inputs):
  """
  Build the Tensorflow graph using the given inputs
  Args:
    graph: tf.Graph instance
    inputs: A dictionary with input placeholders
  """
 
  with graph.as_default():
    ret_dict = tower_loss(inputs)
    return ret_dict


def train_loop(graph, ret_dict, feed_dict):

    tf.logging.set_verbosity(tf.logging.INFO)

    with graph.as_default(), tf.device('/cpu:0'):
	#use the GPUs based on the config
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(opts['gpu_ids'])
        session_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        session_config.gpu_options.allow_growth = True

        mads_unsup = 0
        restore = True
        begin_time = time.time()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        with tf.Session(config=session_config) as sess:

            ######## initialize
            global_init = tf.global_variables_initializer()
            local_init = tf.local_variables_initializer()
            sess.run([global_init,local_init])

            
	    vars_available = [i[0] for i in tf.train.list_variables(tf.train.latest_checkpoint("./log_dir"))]
            ######## restore
            if restore == True:
               try:
                  vars_to_restore = []
                  vars_to_restore1 = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
                  vars_to_restore2 = tf.global_variables()
                  for v in vars_to_restore2:
                      vars_to_restore.append(v)
                  varlist = []
                  for v in vars_to_restore:
                      if "Adam" not in v.name and v.op.name in vars_available:
                          varlist.append(v)
                  restorer = tf.train.Saver(var_list=varlist)
                  restorer.restore(sess, tf.train.latest_checkpoint("./log_dir"))
               except Exception as e:
		  print("error restoring model parameters!!")
		  exit()                 

            ######## create a summary writer:
            summary_writer = tf.summary.FileWriter(opts['log_dir'], graph=sess.graph)

            ######## get the value of the global-step:
	    start_time = time.time()
	    batch_size = opts['batch_size']

	    
	    #used to train on a single train op
	    return_dict = sess.run(ret_dict, feed_dict=feed_dict)
	    duration = time.time() - start_time
	    examples_per_sec = opts['batch_size'] / float(duration)
	    format_str = '%s: step %d, loss_recons = %.4f (%.1f examples/sec) %.3f sec/batch'

	    log_tensorboard(return_dict)

