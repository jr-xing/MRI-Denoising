# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Modified on Feb, 2018 based on the work of jakeret

author: yusun
'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os, sys
import shutil
import math
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

from scadec import util
from scadec.layers import *
from scadec.nets import *

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./"))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./scadec/"))


# from IPython.core.debugger import Tracer
# import ipdb
IFDEBUG = False
def dprint(string):
    global IFDEBUG
    if IFDEBUG:
        print(string)

class Unet_bn(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    :param kwargs: args passed to create_net function. 
    """
   
    def __init__(self, img_channels=3, truth_channels=3, cost_dict_list=[{'name':'l2'}], x_shape = None, y_shape = None,  **kwargs):
        tf.reset_default_graph()

        # basic variables
        self.summaries = kwargs.get("summaries", True)
        self.img_channels = img_channels
        self.truth_channels = truth_channels

        # placeholders for input x and y
        if x_shape != None:
            # Assign shape to use VGGNet - Xing
            self.x = tf.placeholder("float", shape=x_shape)
            self.y = tf.placeholder("float", shape=y_shape)
        else:
            self.x = tf.placeholder("float", shape=[None, None, None, img_channels])
            self.y = tf.placeholder("float", shape=[None, None, None, truth_channels])
            # Added by Xing
            # Offer additional random clear data as style target in computing perceptual loss
            self.yRand = tf.placeholder("float", shape=[None, None, None, truth_channels])
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        # reused variables
        # self.batch_size = tf.shape(self.x)[0]
        self.batch_size = 5
        self.nx = tf.shape(self.x)[1]
        self.ny = tf.shape(self.x)[2]
        self.num_examples = tf.shape(self.x)[0]

        # variables need to be calculated
        #Tracer()
        # self.recons = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
        # Xing
        self.recons = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
        # self.recons, self.dw_h_convs, _ = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
        # self.recons = tf.reshape(unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs),[5,160,160,3])
        #ipdb.set_trace()
        #Tracer()
        # Xing
        self.get_loss_dict = kwargs.get("get_loss_dict", False)
        if self.get_loss_dict:
            self.loss_dict = self._get_cost(cost_dict_list, get_loss_dict=True)
            self.loss = self.loss_dict['total_loss']
            self.valid_loss_dict = self._get_cost(cost_dict_list, get_loss_dict=True)
            self.valid_loss = self.valid_loss_dict['total_loss']
        else:
            self.loss = self._get_cost(cost_dict_list)
            self.valid_loss = self._get_cost(cost_dict_list)
                
        self.avg_psnr = self._get_measure('avg_psnr')
        self.valid_avg_psnr =  self._get_measure('avg_psnr')

    def _get_measure(self, measure):
        total_pixels = self.nx * self.ny * self.truth_channels
        dtype       = self.x.dtype
        flat_recons = tf.reshape(self.recons, [-1, total_pixels])
        flat_truths = tf.reshape(self.y, [-1, total_pixels])

        if measure == 'psnr':
            # mse are of the same length of the truths
            mse = mse_array(flat_recons, flat_truths, total_pixels)
            term1 = log(tf.constant(1, dtype), 10.)
            term2 = log(mse, 10.)
            psnr = tf.scalar_mul(20., term1) - tf.scalar_mul(10., term2)
            result = psnr

        elif measure == 'avg_psnr':
            # mse are of the same length of the truths
            mse = mse_array(flat_recons, flat_truths, total_pixels)
            term1 = log(tf.constant(1, dtype), 10.)
            term2 = log(mse, 10.)
            psnr = tf.scalar_mul(20., term1) - tf.scalar_mul(10., term2)
            avg_psnr = tf.reduce_mean(psnr)
            result = avg_psnr

        else:
            raise ValueError("Unknown measure: "%measure)

        return result
        
    def _get_cost(self, cost_dict_list, get_loss_dict = False):
        """
        Constructs the cost function.

        """
        #Tracer()        
        # dprint('_get_cost')
        total_pixels = self.nx * self.ny * self.truth_channels

        def get_mask(img = None, mode = 'default', img_h = None, img_w = None):
            # Get image shape
            if img != None:
                img_shape = img.get_shape().as_list()

            if img_h != None:
                h = img_h
            else:
                h = img_shape[1]

            if img_w != None:
                w = img_w
            else:
                w = img_shape[2]

            # Generate mask
            if mode == None:
                mask = np.ones([w, h, 1])
            elif mode == 'default':
                mask = np.ones([w, h, 1]) * 0.5
                mask[:,int(w/3):int(2*w/3),:] = 1
                mask[int(h/3):int(2*h/3),int(w/3):int(2*w/3),:] = 1.5
                return mask
            elif mode == 'mid5':
                mask = np.ones([w, h, 1]) * 0.1
                mask[:,int(1*w/5):int(4*w/5),:] = 0.5
                mask[:,int(2*w/5):int(3*w/5),:] = 1.5
                return mask
            elif mode == 'norm':
                from scipy.stats import norm
                scaleX = 1.2
                scaleY = 0.8
                normX = np.linspace(norm.ppf(0.001, scale=scaleX),norm.ppf(0.999,scale=scaleX), img_w)
                normY = np.linspace(norm.ppf(0.001, scale=scaleY),norm.ppf(0.999,scale=scaleY), img_h)
                normXX, normYY = np.meshgrid(normX, normY)
                maskN = norm.pdf(np.abs(normXX)+np.abs(normYY))
                maskN = (maskN-np.min(maskN))/np.max(maskN)
                return np.float32(maskN)

        def get_mask_tf(img = None, mode = 'default'):
            # https://stackoverflow.com/questions/39157723/how-to-do-slice-assignment-in-tensorflow
            # p1 = tf.placeholder(tf.float32, [None,5,5,3])
            # mr1 = tf.Variable(tf.ones([5,5,1]), trainable = False)
            # mr2 = mr1[:3,:3,:].assign(tf.ones([3,3,1])*5)
            # init = tf.global_variables_initializer()
            # with tf.Session() as sess:
            #     sess.run(init)
            #     print('VALUE:')
            #     print(sess.run(mr1))
                
            # Get image shape            
            img_shape = img.get_shape().as_list()
            h = img_shape[1]        
            w = img_shape[2]

            # Generate mask
            if mode == None:
                mask = tf.ones([w, h, 1])
            elif mode == 'default':
                mask = tf.ones([w, h, 1]) * 0.5
                mask = mask[:,int(w/3):int(2*w/3),:].assign(tf.ones([h,int(2*w/3)-int(w/3),1])*1)
                mask = mask[:,int(w/3):int(2*w/3),:].assign(tf.ones([int(2*h/3)-int(h/3),int(2*w/3)-int(w/3),1])*1.5)
                return mask
            elif mode == 'mid5':
                mask = tf.ones([w, h, 1]) * 0.1
                mask = mask[:,int(1*w/5):int(4*w/5),:].assign(tf.ones([h,int(4*w/5)-int(1*w/5),1])*0.5)
                mask = mask[:,int(2*w/5):int(3*w/5),:].assign(tf.ones([h,int(3*w/5)-int(2*w/5),1])*1.5)
                return mask
            elif mode == 'norm':
                from scipy.stats import norm
                scaleX = 1.2
                scaleY = 0.8
                normX = np.linspace(norm.ppf(0.001, scale=scaleX),norm.ppf(0.999,scale=scaleX), img_w)
                normY = np.linspace(norm.ppf(0.001, scale=scaleY),norm.ppf(0.999,scale=scaleY), img_h)
                normXX, normYY = np.meshgrid(normX, normY)
                maskN = norm.pdf(np.abs(normXX)+np.abs(normYY))
                maskN = (maskN-np.min(maskN))/np.max(maskN)
                # return np.float32(maskN)
                return tf.constant(maskN)
        
        # def get_grad_old(img):
        #     dify = img[:,  1:, :-1,:] - img[:,:-1,:-1,:]
        #     difx = img[:, :-1, 1:, :] - img[:,:-1,:-1,:]
        #     return dify+difx

        # def get_grad(img):
        #     difX = tf.reshape(tf.constant([[1,0,-1],[1,0,-1],[1,0,-1]],tf.float32),[3, 3, 1, 1])
        #     difY = tf.reshape(tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]],tf.float32),[3, 3, 1, 1])
        #     gradX = tf.nn.conv2d(img, sobelX, strides=[1,1,1,1], padding='SAME')[:,1:-1,1:-1,:]
        #     gradY = tf.nn.conv2d(img, sobelY, strides=[1,1,1,1], padding='SAME')[:,1:-1,1:-1,:]
        #     dify = img[:,  1:, :-1,:] - img[:,:-1,:-1,:]
        #     difx = img[:, :-1, 1:, :] - img[:,:-1,:-1,:]
        #     return tf.sqrt(tf.square(difx)+tf.square(dify))
                
        
        # def get_Sobel(img):
        #     sobelX = tf.reshape(tf.constant([[1,0,-1],[2,0,-2],[1,0,-1]],tf.float32),[3, 3, 1, 1])
        #     sobelY = tf.reshape(tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]],tf.float32),[3, 3, 1, 1])
        #     imgX = tf.nn.conv2d(img, sobelX, strides=[1,1,1,1], padding='SAME')[:,1:-1,1:-1,:]
        #     imgY = tf.nn.conv2d(img, sobelY, strides=[1,1,1,1], padding='SAME')[:,1:-1,1:-1,:]
        #     return tf.sqrt(tf.square(imgX)+tf.square(imgY))

        def get_edge(img, operator, get_XY = False):
            # https://blog.csdn.net/huanghuangjin/article/details/81130171
            # https://www.cnblogs.com/wxl845235800/p/7700867.html
            if operator == 'gradient':
                gradX = tf.reshape(tf.constant([[0,0,0],[1,0,-1],[0,0,0]],tf.float32),[3, 3, 1, 1])
                gradY = tf.reshape(tf.constant([[0,1,0],[0,0,0],[0,-1,0]],tf.float32),[3, 3, 1, 1])
                imgX = tf.nn.conv2d(img, gradX, strides=[1,1,1,1], padding='SAME')
                imgY = tf.nn.conv2d(img, gradY, strides=[1,1,1,1], padding='SAME')
                if get_XY:
                    return imgX, imgY
                else:
                    return tf.sqrt(tf.square(imgX)+tf.square(imgY))

            elif operator == 'Sobel' or operator == 'sobel':
                sobelX = tf.reshape(tf.constant([[1,0,-1],[2,0,-2],[1,0,-1]],tf.float32),[3, 3, 1, 1])
                sobelY = tf.reshape(tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]],tf.float32),[3, 3, 1, 1])
                imgX = tf.nn.conv2d(img, sobelX, strides=[1,1,1,1], padding='SAME')
                imgY = tf.nn.conv2d(img, sobelY, strides=[1,1,1,1], padding='SAME')
                if get_XY:
                    return imgX, imgY
                else:
                    return tf.sqrt(tf.square(imgX)+tf.square(imgY))

            elif operator == 'LoG':
                # Laplacian of Gaussian (LoG)
                # http://academic.mu.edu/phys/matthysd/web226/Lab02.htm
                LoG = tf.reshape(tf.constant([[0,-1,0],[-1,4,-1],[0,-1,0]],tf.float32),[3, 3, 1, 1])
                return tf.nn.conv2d(img, LoG, strides=[1,1,1,1], padding='SAME')


        loss = 0
        loss_dict = {}
        for cost_dict in cost_dict_list:
            if cost_dict['name'] == 'l2' or cost_dict['name'] == 'mean_square_error':                
                mask = get_mask(mode=cost_dict.get('mask',None),img_h=320,img_w=320)                
                loss_l2 = tf.losses.mean_squared_error(tf.multiply(self.recons,mask), tf.multiply(self.y,mask))
                current_loss = loss_l2
                current_loss_name = 'mean_square_error'
                #loss += cost_dict['weight']*loss_l2
                #loss_dict['meaｎ_square_loss'] = loss_l2

            elif cost_dict['name'] == 'edge':
                mask = get_mask(mode=cost_dict.get('mask',None),img_h=320,img_w=320)
                if cost_dict.get('mask_before_operate',False):
                    # Get masked images
                    recons_masked = tf.multiply(self.recons, mask)
                    y_masked = tf.multiply(self.y, mask)

                    # Compute edge loss
                    if cost_dict.get('get_XY',False):
                        edge_recons_X,edge_recons_Y = get_edge(recons_masked, operator=cost_dict['edge_type'], get_XY=True)
                        edge_y_X,edge_y_Y = get_edge(y_masked, operator=cost_dict['edge_type'], get_XY=True)
                        loss_masked_edge = tf.losses.absolute_difference(edge_recons_X, edge_y_X)+tf.losses.absolute_difference(edge_recons_Y, edge_y_Y)
                    else:
                        edge_recons = get_edge(recons_masked, operator=cost_dict['edge_type'])
                        edge_y = get_edge(y_masked, operator=cost_dict['edge_type'])
                        loss_masked_edge = tf.losses.mean_squared_error(edge_recons, edge_y)   
                    # Loss
                    current_loss = loss_masked_edge
                    current_loss_name = cost_dict['edge_type']

                else:                    
                    if cost_dict.get('get_XY',False):
                        edge_recons_X,edge_recons_Y = get_edge(self.recons, operator=cost_dict['edge_type'], get_XY=True)
                        edge_y_X,edge_y_Y = get_edge(self.y, operator=cost_dict['edge_type'], get_XY=True)
                        loss_edge_masked = tf.losses.absolute_difference(tf.multiply(edge_recons_X,mask), tf.multiply(edge_y_X,mask)) + \
                                            tf.losses.absolute_difference(tf.multiply(edge_recons_Y,mask), tf.multiply(edge_y_Y,mask))
                    else:
                        edge_recons = get_edge(self.recons, operator=cost_dict['edge_type'])
                        edge_y = get_edge(self.y, operator=cost_dict['edge_type'])
                        loss_edge_masked = tf.losses.mean_squared_error(tf.multiply(edge_recons, mask), tf.multiply(edge_y, mask))
                    current_loss = loss_edge_masked
                    current_loss_name = cost_dict['edge_type']

            else:
                raise ValueError("Unknown cost function: "+cost_dict['name'])

            if cost_dict.get('upper_bound',False):
                #current_loss =  tf.clip_by_value(current_loss, 0.0, cost_dict.get('upper_bound',0.5))
                current_loss =  tf.clip_by_norm(current_loss, cost_dict.get('upper_bound',0.5))
            
            loss += cost_dict['weight']*current_loss
            loss_dict[current_loss_name] = current_loss

        loss_dict['total_loss'] = loss    


        
        
        if get_loss_dict:
            return loss_dict
        else:
            return loss
    


    # predict
    def predict(self, model_path, x_test, keep_prob, phase):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            prediction = sess.run(self.recons, feed_dict={self.x: x_test, 
                                                          self.keep_prob: keep_prob, 
                                                          self.phase: phase})  # set phase to False for every prediction
                            # define operation
        return prediction
    
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)
