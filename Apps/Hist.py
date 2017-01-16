#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:03:21 2016

@author: ful6ru04
"""
from __future__ import print_function

import time
import numpy as np
import gevent

from gevent.pool import Pool
from multiprocessing.pool import ThreadPool

class Hist():
    
    # Histogram of intensities distribution of a image.
    # Input: A image with the size of high-by-width.
    # Output: A histogram with size of 1-by-bins.
    @staticmethod
    def histDist( image ):
        bins=4096
        image = np.squeeze( image )
        hist = np.zeros( [1, bins] )

        for x in np.nditer( image ):
            if ( x >= 0 ):
                hist[0,int(x)] = hist[0,int(x)] + 1
        
        return hist

    # Histograms of intensities distribution of a set of images.
    # Input: A image with the size of batch-by-high-by-width.
    # Output: A histogram with size of batch-by-bins.
    @staticmethod
    def batchHistDist( batch_imgs, bins ):
        start_time = time.time()
        if ( batch_imgs.shape.__len__()<=2 ):
            raise ValueError("Error! The rank of the input array must be 3!")

        hist = np.empty([ batch_imgs.shape[0], bins ])
        pool = Pool(8)
        
        for img in range( batch_imgs.shape[0] ):
            #hist[img,:] = 
            re = pool.map( Hist.histDist, batch_imgs[img,:,:] )
            hist[img,:] = list(re)
            #hist[img,:] = Hist.histDist( np.squeeze(batch_imgs[img,:,:]),bins )
        
            
        elapsed_time = time.time() - start_time
        print('Done!', end='')
        print(' (Elasped time: {:02.2f}'.format(elapsed_time)+'s)')
        
        return hist
        
    @staticmethod
    def batchHistDistThread( batch_imgs, bins ):
        
        if ( batch_imgs.shape.__len__()<=2 ):
            raise ValueError("Error! The rank of the input array must be 3!")
        start_time = time.time()
        hist = np.empty([ batch_imgs.shape[0], bins ])
        pool = ThreadPool(processes=10)
        
        async_result = pool.apply_async(Hist.batchHistDist,(batch_imgs,bins))
        return_val = async_result.get()
        hist = return_val
            

        
        elapsed_time = time.time() - start_time
        print('Done!', end='')
        print(' (Elasped time: {:02.2f}'.format(elapsed_time)+'s)')
            
        return hist
        
    # Extract histograms of intensities distribution from every pixel position.
    # Input: A image with the size of batch-by-high-by-width [b,h,w].
    # Output: A histogram with size of batch-by-high-by-width-by-bins [b,h,w,bins].
    @staticmethod
    def extractHistDist( batch_imgs, bins, console=True ): # batch_imgs: [batch,height,width]
        
        if (console):
            print('Extracting patch-wise histograms ... ', end='')
        start_time = time.time()
    
        patch_sizes = [ 3, 5, 7 ]
        patch_sizes.sort()
        
        if not ( batch_imgs.shape.__len__()==3 ):
            raise ValueError("Error! Element errors in patch array!")
        if ( patch_sizes.__len__()<1 ):
            raise ValueError("Error! Element errors in patch array!")
        for ind in range( patch_sizes.__len__() ):
            if ( patch_sizes[ind]%2==0 ):
                raise ValueError("Error! The patch size must be odd!")

        # maximum padding size
        mx = np.max(patch_sizes) - 1 # minus centre
        pad_batch_imgs = np.pad(batch_imgs,((0,0),(mx,mx),(mx,mx)),'reflect')
        
        dense_hist = np.empty([ batch_imgs.shape[0], batch_imgs.shape[1], 
                           batch_imgs.shape[2], bins ])
        for r in range( mx,batch_imgs.shape[1]-mx ): # row wise
            for c in range( mx,batch_imgs.shape[2]-mx ): # col wise
                
                patch = pad_batch_imgs[ :, r-mx:r+mx, c-mx:c+mx ]
                hist = Hist.batchHistDist(patch,bins)
                hist_reshape = np.reshape(hist, (hist.shape[0],1,1,hist.shape[1]))
                dense_hist[:,r:r+1,c:c+1,:] = hist_reshape
                
        if (console):
            elapsed_time = time.time() - start_time
            print('Done!', end='')
            print(' (Elasped time: {:02.2f}'.format(elapsed_time)+'s)')
        
        return dense_hist
        
        
        
        
        
