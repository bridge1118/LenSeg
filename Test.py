#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:34:34 2016

@author: ful6ru04

This is testing phase

"""

import tensorflow as tf
import os

import scipy.io
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

from Apps.AppIm import MyClassImages as Cim
from Apps.AppIm import Read
from Apps.MyGraph import MyGraph

############################## Variables ##############################
# Graph data
data_shape = (1,512,512,1) # [batch_size,h,w,c]

# Testing variables
output_dir = './output'
imdir_test  = './DICOM_dat4'

# Checkpoint
ckpt_dir = './ckpt'
ckpt_name = 'net.ckpt'

############################## Initialize ##############################
# Initialize data
im_test = Cim(imdir_test,data_shape)

# Graph variables
xs = tf.placeholder(tf.float32,data_shape,name='xs')
train = tf.placeholder(tf.bool)
global_step = tf.Variable(0, name='global_step', trainable=False)

# Build Graph
myGraph = MyGraph()
cross_entropy = myGraph.FCN(xs,train)
prediction, prob = myGraph.predict(cross_entropy)

# Run Graph
sess = tf.Session()

# Restore weights from checkpoint
saver = tf.train.Saver()
if not os.path.exists( os.path.join(ckpt_dir,'checkpoint') ):
    raise ValueError("Error! There is no learned weights! Please train the graph first!")
    
print('Restoring graph from last checkpoint!')
saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
step = sess.run([global_step],feed_dict={})
print('Running graph using the weight which are trained after '+
      str(step[0])+' iteration training phase!')

############################## Run Graph ##############################
# Make output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count_batch = 0
batch_xs = np.empty( data_shape )
step = 0

for x in im_test:

    print( 'Image no.: ' + str(step) )
    step = step + 1
    batch_xs[count_batch,:,:,:] = x
    count_batch = count_batch + 1

    if ( count_batch==data_shape[0] ):
        pred = sess.run(prediction,feed_dict={xs:batch_xs, train:False})
        pro = sess.run(prob,feed_dict={xs:batch_xs, train:False})
        
        
        for batch in range(data_shape[0]):
            img = np.squeeze(batch_xs[batch,:,:])
            scipy.misc.imsave(output_dir+'/test'+str(step*data_shape[0]+0)+'_in'+'.jpg', img)
            
            pr = np.squeeze(pred[batch,:,:])
            scipy.misc.imsave(output_dir+'/test'+str(step*data_shape[0]+0)+'_out'+'.jpg', pr)
            
            pro = np.squeeze(pro[batch,:,:,1])
            plt.imsave(output_dir+'/test'+str(step*data_shape[0]+0)+'_pr'+'.jpg', pro)
            
        
        count_batch = 0
        
sess.close()

