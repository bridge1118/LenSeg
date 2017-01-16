#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:41:42 2016

@author: ful6ru04
"""

import tensorflow as tf

from Apps.MyNet import MyNet as net


class MyGraph():
    
    def __init__(self):
        
        self.weights_size = {        # [k,k, in,out]
                             'conv1_1':[1,1,  1,200],
                             'conv2_1':[1,1,200, 20],
                             'conv3_1':[1,1, 20,  2]}

    def FCN(self,xs,train):
        ########## LAYER DEFINITION ##########
        ### layer 1
        conv1_1 = net.conv_layer(     xs, self.weights_size['conv1_1'], name='conv1_1')
        
        conv2_1 = net.conv_layer(conv1_1, self.weights_size['conv2_1'], name='conv2_1')
        
        conv3_1 = net.conv_layer(conv2_1, self.weights_size['conv3_1'], name='conv3_1')
        
        cross_entropy = conv3_1
        
        return cross_entropy
        
    def predict(self,cross_entropy):
        prob = tf.nn.softmax(cross_entropy)
        prediction = tf.argmax(cross_entropy, dimension=3)
        return prediction, prob
        
    def FCN_train(self,cross_entropy,ys,global_step,learning_rate=1e-4):
        
        # training solver
        with tf.name_scope('loss'):
           
            ys_reshape = tf.squeeze(ys,squeeze_dims=[3])
            
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(cross_entropy,ys_reshape)
            cross_entropy = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss',cross_entropy)
            
        with tf.name_scope('solver'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

        return train_step, cross_entropy
    
