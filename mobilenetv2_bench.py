import sys
sys.path.append("/home/mikeliao/workspace/models/research/slim")
import time
import numpy as np
import os
import glob
import tensorflow as tf
import math
from datetime import datetime

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from nets.mobilenet import mobilenet_v2
from tensorflow.contrib import slim

BATCH_SIZE = 1
NUM_FILES = 25 

image_size = 224

with tf.Graph().as_default():
    images = tf.Variable(tf.random_normal([1, image_size, image_size, 3],
                                                    dtype=tf.float32,
                                                    stddev=1e-1))
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        logits, endpoints = mobilenet_v2.mobilenet(images)
    
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    #config = tf.ConfigProto(gpu_options=gpu_options)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True   
    
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    time_burn_in = 5
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        #Add code for writer
        writer = tf.summary.FileWriter(logdir='tensorboard/mobilenetv2', graph=sess.graph) 

        total_duration = 0.0
        total_duration_squared = 0.0
        for i in range(NUM_FILES):
            start = time.time()
            prob = sess.run([endpoints['Predictions']], options=run_options, 
                                                        run_metadata=run_metadata)

            duration = (time.time() - start) * 1000
            writer.add_run_metadata(run_metadata, "step {}".format(i))

            if (i > time_burn_in):
                total_duration += duration
                total_duration_squared += duration * duration
                
        num_images = NUM_FILES - time_burn_in
        mn = total_duration / (num_images)
        vr = total_duration_squared / num_images - mn * mn
        sd = math.sqrt(vr)
        print ('%s: across %d images, %.3f +/- %.3f millisec / image' %
             (datetime.now(), num_images, mn, sd))
            
