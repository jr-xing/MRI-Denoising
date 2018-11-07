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

from scadec_Kraken import util
from scadec_Kraken.layers import *
from scadec_Kraken.nets_Kraken import *

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./"))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./scadec/"))


# from IPython.core.debugger import Tracer
# import ipdb
IFDEBUG = False
# def dprint(string):
#     global IFDEBUG
#     if IFDEBUG:
#         print(string)

def get_highPass(img, mode='gradient', paras = {}):
    if mode == 'gradient':
        imgY, imgX = tf.image.image_gradients(img)
        return tf.sqrt(tf.square(imgX)+tf.square(imgY), name='xHighPass')
    elif mode == 'fft':                        
        # n, h, w, c = tf.shape(img)
        n = 5;h = 320; w = 320; c = 1;
        freq_thershold = paras.get('freq_thershold', int(h/3))
        img_fft = tf.fft2d(tf.cast(img,tf.complex64))
        
        left = freq_thershold;   right = w - freq_thershold
        up = freq_thershold;     down = h - freq_thershold
        mask = np.ones([h, w, c])
        # mask[up:down, left:right, :] = 0
        # return tf.real(tf.ifft(tf.multiply(img_fft, mask)))
        return tf.real(tf.ifft(tf.fft(tf.cast(img, tf.complex64))))

def get_lowPass(img, mode='average', paras = {}):
    if mode =='average':
        size = paras.get('size', 7)
        lowPassFilter_C3 = tf.constant(1/size**2, shape=[size, size, 3, 3], name='lowPass_filter_C1')
        return tf.nn.conv2d(img, lowPassFilter_C3, strides = [1,1,1,1], padding='SAME', name = 'xlowPass')
    elif mode == 'fft':
        # n, h, w, c = img.shape
        n = 5;h = 320; w = 320; c = 1;
        freq_thershold = paras.get('freq_thershold', int(h/3))
        img_fft = tf.fft2d(tf.cast(img,tf.complex64))
        
        left = freq_thershold;   right = w - freq_thershold
        up = freq_thershold;     down = h - freq_thershold
        mask = np.ones([h, w, c])
        # mask = np.zeros([h, w, c])
        # mask[up:down, left:right, :] = 1
        # return tf.real(tf.ifft(tf.multiply(img_fft, mask)))
        return tf.real(tf.ifft(tf.fft(tf.cast(img, tf.complex64))))

class Unet_bn(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    :param kwargs: args passed to create_net function. 
    """
   
    # def __init__(self, img_channels=3, truth_channels=3, cost_dict_list=[{'name':'l2'}], x_shape = None, y_shape = None, **kwargs):
    def __init__(self, kwargs_list, img_channels=3, truth_channels=3, cost_dict_lists=[["naive",{'name':'l2'}]], x_shape = None, y_shape = None):
        tf.reset_default_graph()

        # basic variables
        kwargs = kwargs_list[0]
        self.structure_type = kwargs.get('structure_type','highLowPass')
        self.summaries = kwargs.get("summaries", True)
        self.img_channels = img_channels
        self.truth_channels = truth_channels
        self.batch_size = kwargs.get("batch_size", 5)
        self.ifGAN = kwargs.pop('GAN', False)

        # placeholders for input x and y    
        # self.x = tf.placeholder("float", shape=x_shape, name='x')
        # self.y = tf.placeholder("float", shape=y_shape, name='y')
        self.x = tf.placeholder("float", shape=[None, None, None, img_channels], name='x')
        self.y = tf.placeholder("float", shape=[None, None, None, truth_channels], name='y')
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)
        self.current_epoch = tf.placeholder(tf.int32, name='current_epoch')
        self.total_epochs = tf.placeholder(tf.int32, name='total_epochs')


        # reused variables
        self.nx = tf.shape(self.x)[1]
        self.ny = tf.shape(self.x)[2]
        self.num_examples = tf.shape(self.x)[0]

        # variables need to be calculated
        # self.recons = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
        # Xing
        # Kraken change
        # self.structure = kwargs.get('structure',{'type':'Hydra'})
        # if type(self.structure) == dict:
        #     self.structure_type = self.structure['type']
        #     self.batch_cls = tf.placeholder(tf.int32,[None, kwargs['structure'].get('n_classes',1)], name='batch_cls')
        # else:
        #     self.structure_type = self.structure
        #     self.batch_cls = tf.placeholder(tf.int32,[None, kwargs.get('n_classes',1)], name='batch_cls')
        # self.structure_type = 'Hydra'
        self.structure_type = kwargs.get('structure_type','highLowPass')
        self.batch_cls = tf.placeholder(tf.int32,[None, kwargs.get('n_classes',1)], name='batch_cls')
        # self.batch_cls = 1
        
        self._get_net(kwargs_list)        

        # Xing
        self.loss_dict = self._get_cost(cost_dict_lists)
        self.valid_loss_dict = self._get_cost(cost_dict_lists)
                
        # Total loss        
        self.loss = self.loss_dict['total_loss']
        self.valid_loss = self.valid_loss_dict['total_loss']
                
        self.avg_psnr = self._get_measure('avg_psnr')
        self.valid_avg_psnr =  self._get_measure('avg_psnr')

    def _get_net(self, kwargs_list):
        # def get_lowPass(mode='constant'):
        #     if mode='constant':
        #         return 
        
        kwargs = kwargs_list[0]
        if len(kwargs_list) > 1:
            structure_type = kwargs.pop('structure_type','highLowPass')
            if structure_type == 'highLowPass':
                lowPass_kwargs  = [kwargs for kwargs in kwargs_list[1:] if kwargs['name']=='lowPass'][0]
                highPass_kwargs = [kwargs for kwargs in kwargs_list[1:] if kwargs['name']=='highPass'][0]
                if lowPass_kwargs['structure']['type'] != 'Nagini':
                    raise ValueError("Not supported structure: %s" % lowPass_kwargs['structure'])
                if highPass_kwargs['structure']['type'] != 'Nagini':
                    raise ValueError("Not supported structure: %s" % highPass_kwargs['structure'])
                
                # lowPassFilter_np = np.ones((3,3))/9
                # lowPassFilter_tf = tf.constant_initializer(value=lowPassFilter_np, dtype=tf.float32)
                

                
                # self.xLowPass = tf.nn.conv2d(self.x, self.lowPassFilter_C3, strides = [1,1,1,1], padding='SAME', name = 'xlowPass')
                # https://datascience.stackexchange.com/questions/19945/tensorflow-can-not-convert-float-into-a-tensor
                # self.xHighPass = tf.subtract(self.x, self.xLowPass, name = 'xHighPass')

                # self.xLowPass = get_lowPass(self.x, paras = {'size':7})
                self.xLowPass = get_lowPass(self.x, mode='fft')
                # print('XXXXXXXXXXXXXXXXXXXXXXX')
                # print(self.x)
                # print(self.x.shape)
                self.xHighPass = get_highPass(self.x, mode='fft')
                # self.xHighPass = self.x
                # imgY, imgX = tf.image.image_gradients(self.x)
                # self.xHighPass = tf.sqrt(tf.square(imgX)+tf.square(imgY), name='xHighPass')
                with tf.variable_scope('lowPass'):
                    self.recons_lowPass  = unet_decoder(self.xLowPass,  self.keep_prob, self.phase, self.img_channels, self.truth_channels, **{**kwargs,**lowPass_kwargs})
                with tf.variable_scope('highPass'):
                    self.recons_highPass = unet_decoder(self.xHighPass, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **{**kwargs,**highPass_kwargs})                                
                
                
                # self.recons = (self.recons_lowPass + self.recons_lowPass)/2
                recons_concat = concat(self.recons_highPass, self.recons_lowPass)#,'recons_concat')
                recons_concat = conv2d(recons_concat, 3, 10, self.keep_prob, 'recons_1')
                recons_concat = conv2d(recons_concat, 3, 3, self.keep_prob, 'recons_2')
                self.recons = conv2d(recons_concat, 1, self.truth_channels, self.keep_prob, 'recons')
                # output = conv2d(in_node, 1, truth_channels, keep_prob, '{}_conv2truth_channels'.format(neck_idx))

            else:
                raise ValueError('Unknown structure_type: %s' % structure_type)
                    
        else:
            if kwargs.pop('no_GAN_net_func', False):
                if self.structure_type == 'Hydra' or self.structure_type == 'HydraEr':
                    self.necks = unet_decoder_noGAN(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
                elif self.structure_type == 'Nagini':
                    self.recons = unet_decoder_noGAN(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
                else:
                    raise ValueError('Unknown Net Structure: '+self.structure_type)        
            else:
                if self.structure_type == 'Hydra' or self.structure_type == 'HydraEr':
                    self.necks = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
                elif self.structure_type == 'Nagini':
                    self.recons = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
                else:
                    raise ValueError('Unknown Net Structure type: '+self.structure_type)        
    
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
        
    def _get_cost(self, cost_dict_lists):
        """
        Constructs the cost function.

        """
        #Tracer()        
        dprint('Computing Cost...')
        # total_pixels = self.nx * self.ny * self.truth_channels

        def get_mask(mode = 'default', h = 320, w = 320, paras = {}):
            # Generate mask
            if mode == None:
                return np.float32(np.ones([w, h, 1]))                
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
                scaleX = paras.get('scaleX', 1.2)
                scaleY = paras.get('scaleY', 0.8)
                normX = np.linspace(norm.ppf(0.001, scale=scaleX),norm.ppf(0.999,scale=scaleX), w)
                normY = np.linspace(norm.ppf(0.001, scale=scaleY),norm.ppf(0.999,scale=scaleY), h)
                normXX, normYY = np.meshgrid(normX, normY)
                maskN = norm.pdf(np.abs(normXX)+np.abs(normYY))
                maskN = (maskN-np.min(maskN))/np.max(maskN)
                return np.float32(np.reshape(maskN,[w,h,1]))     
            elif mode == 'GaussianHighPass':
                normMask = get_mask('norm', h, w)
                return 1 - normMask

        def get_edge(img, operator, get_XY = False, NMS = False, NMS_window_size = 3):
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
            else:
                raise ValueError("Unknown edge type: "+operator)

        
        def get_losses(x, y, cost_dict_list, prefix = ''):
            loss = 0
            loss_dict = {}
            for cost_dict in cost_dict_list:

                if cost_dict['name'] == 'l2' or cost_dict['name'] == 'mean_square_error':                
                    mask = get_mask(mode=cost_dict.get('mask',None),h=320,w=320)
                    loss_l2 = tf.losses.mean_squared_error(tf.multiply(x,mask), tf.multiply(y,mask))
                    current_loss = loss_l2
                    current_loss_name = cost_dict['name']

                elif cost_dict['name'] == 'edge':
                    mask = get_mask(mode=cost_dict.get('mask',None),h=320,w=320)
                    if cost_dict.get('mask_before_operate',False):
                        # Get masked images
                        x_masked = tf.multiply(x, mask)
                        y_masked = tf.multiply(y, mask)

                        # Compute edge loss
                        if cost_dict.get('get_XY',False):
                            edge_x_X,edge_x_Y = get_edge(x_masked, operator=cost_dict['edge_type'], get_XY=True, NMS=cost_dict.get('NMS', False), NMS_window_size=cost_dict.get('NMS_window_size', 3))
                            edge_y_X,edge_y_Y = get_edge(y_masked, operator=cost_dict['edge_type'], get_XY=True, NMS=cost_dict.get('NMS', False), NMS_window_size=cost_dict.get('NMS_window_size', 3))
                            typ = cost_dict.get('type', None)
                            if typ == None:
                                loss_masked_edge = tf.losses.absolute_difference(edge_x_X, edge_y_X)+tf.losses.absolute_difference(edge_x_Y, edge_y_Y)
                            elif typ == '2':
                                loss_masked_edge = tf.losses.mean_squared_error(edge_x_X, edge_y_X) + tf.losses.mean_squared_error(edge_x_Y, edge_y_Y)
                            elif typ == '3':
                                edge_x = tf.square(edge_x_X) + tf.square(edge_x_Y)
                                edge_y = tf.square(edge_y_X) + tf.square(edge_y_Y)
                                loss_masked_edge = tf.losses.mean_squared_error(edge_x, edge_y)
                        else:
                            edge_x = get_edge(x_masked, operator=cost_dict['edge_type'])
                            edge_y = get_edge(y_masked, operator=cost_dict['edge_type'])
                            loss_masked_edge = tf.losses.mean_squared_error(edge_x, edge_y)                           
                        # Loss
                        current_loss = loss_masked_edge
                        current_loss_name = cost_dict['edge_type']

                    else:                    
                        if cost_dict.get('get_XY',False):
                            edge_x_X,edge_x_Y = get_edge(x, operator=cost_dict['edge_type'], get_XY=True)
                            edge_y_X,edge_y_Y = get_edge(y, operator=cost_dict['edge_type'], get_XY=True)
                            loss_edge_masked = tf.losses.absolute_difference(tf.multiply(edge_x_X,mask), tf.multiply(edge_y_X,mask)) + \
                                               tf.losses.absolute_difference(tf.multiply(edge_x_Y,mask), tf.multiply(edge_y_Y,mask))
                        else:
                            edge_x = get_edge(x, operator=cost_dict['edge_type'])
                            edge_y = get_edge(y, operator=cost_dict['edge_type'])
                            loss_edge_masked = tf.losses.mean_squared_error(tf.multiply(edge_x, mask), tf.multiply(edge_y, mask))
                        current_loss = loss_edge_masked
                        current_loss_name = cost_dict['edge_type']

                    current_loss = tf.cond(tf.less_equal(self.current_epoch, self.total_epochs-cost_dict.get('invalid_last', 0)), lambda:current_loss, lambda:0.0)
                
                elif cost_dict['name'] == 'kl2':
                    mask = get_mask(mode=cost_dict.get('mask',None),h=320,w=320)
                    x_fft = tf.multiply(tf.fft2d(tf.cast(x,tf.complex64)), mask)
                    y_fft = tf.multiply(tf.fft2d(tf.cast(y,tf.complex64)), mask)
                    loss_kl2 = tf.losses.mean_squared_error(tf.real(x_fft), tf.real(y_fft)) + tf.losses.mean_squared_error(tf.imag(x_fft), tf.imag(y_fft))
                    current_loss = loss_kl2
                    current_loss_name = cost_dict['name']
                
                elif cost_dict['name'] == 'kl1':
                    mask = get_mask(mode=cost_dict.get('mask',None),h=320,w=320)
                    x_fft = tf.multiply(tf.fft2d(tf.cast(x,tf.complex64)), mask)
                    y_fft = tf.multiply(tf.fft2d(tf.cast(y,tf.complex64)), mask)
                    loss_kl1 = tf.losses.absolute_difference(tf.real(x_fft), tf.real(y_fft)) + tf.losses.absolute_difference(tf.imag(x_fft), tf.imag(y_fft))
                    current_loss = loss_kl1
                    current_loss_name = cost_dict['name']    
                
                else:
                    raise ValueError("Unknown cost function: "+cost_dict['name'])

                if cost_dict.get('upper_bound',False):
                    #current_loss =  tf.clip_by_value(current_loss, 0.0, cost_dict.get('upper_bound',0.5))
                    current_loss =  tf.clip_by_norm(current_loss, cost_dict.get('upper_bound',0.5))
                
                
                loss += cost_dict['weight']*current_loss
                loss_dict[prefix + current_loss_name] = current_loss
                #loss +=  tf.cond(check_if_valid(cost_dict), lambda: cost_dict['weight']*current_loss, lambda: 0.0)                
                #loss_dict[current_loss_name] = tf.cond(check_if_valid(cost_dict), lambda: current_loss, lambda: 0.0)

            loss_dict['total_loss'] = loss
            return loss_dict        
        
        # Initial loss dict
        # Create total loss dict
        # EDIT!
        total_loss_dict = {}
        # total_loss_dict['total_loss'] = 0
        # for cost_dict in cost_dict_list:
        #     if cost_dict['name'] == 'edge':
        #         total_loss_dict[cost_dict['edge_type']]=0
        #     else:
        #         total_loss_dict[cost_dict['name']] = 0
        
        # Get loss
        if self.structure_type == 'highLowPass':  
            # self.lowPassFilter = tf.constant(1/9, shape=[1, 3, 3, 1], name='lowPass_filter')
            # xLowPass = tf.nn.conv2d(self.x, self.lowPassFilter, strides = [1,1,1,1], padding='SAME', name = 'lowPass')

            # def get_gradient(img):            
            #     imgY, imgX = tf.image.image_gradients(img)
            #     return tf.sqrt(tf.square(imgX)+tf.square(imgY), name='yHighPass')
            
            # def get_blured(img, size = 3):
            #     self.lowPassFilter_C1 = tf.constant(1/size**2, shape=[size, size, 1, 1], name='lowPass_filter_C1')
            #     return tf.nn.conv2d(img, self.lowPassFilter_C1, strides = [1,1,1,1], padding='SAME', name = 'ylowPass')

            # self.lowPassFilter_C1 = tf.constant(1/9, shape=[3, 3, 1, 1], name='lowPass_filter_C3')
            # yLowPass = tf.nn.conv2d(self.y, self.lowPassFilter_C1, strides = [1,1,1,1], padding='SAME', name = 'ylowPass')
            self.yLowPass = get_lowPass(self.y, mode='fft')
            lowPass_loss_list = [cost_dict_list for cost_dict_list in cost_dict_lists if cost_dict_list[0]=='forLowPass'][0][1:]
            lowPass_loss_dict = get_losses(self.recons_lowPass, self.yLowPass, lowPass_loss_list, prefix='lowPass')
            
            # yHighPass = tf.subtract(self.y, yLowPass, name = 'yHighPass')
            # print('YYYYYYYYYYYYYYYYYYYYYYY')
            # print(self.y)
            self.yHighPass = get_highPass(self.y, mode='fft')
            highPass_loss_list = [cost_dict_list for cost_dict_list in cost_dict_lists if cost_dict_list[0]=='forHighPass'][0][1:]
            highPass_loss_dict = get_losses(self.recons_highPass, self.yHighPass, highPass_loss_list, prefix='highPass')
            # print(highPass_loss_dict.keys())
            
            recons_loss_list = [cost_dict_list for cost_dict_list in cost_dict_lists if cost_dict_list[0]=='forRecon'][0][1:]
            recons_loss_dict = get_losses(self.recons, self.y, recons_loss_list, prefix='recon')

            for loss_dict in [lowPass_loss_dict, highPass_loss_dict, recons_loss_dict]:
                for key, value in loss_dict.items():
                    if key in total_loss_dict:
                        total_loss_dict[key] += value
                    else:
                        total_loss_dict[key]  = value
            
            # for key, value in highPass_loss_dict.items():
            #     if key in total_loss_dict:
            #         total_loss_dict[key] += value
            #     else:
            #         total_loss_dict[key]  = value
            
            # for key, value in highPass_loss_dict.items():
            #     if key in total_loss_dict:
            #         total_loss_dict[key] += value
            #     else:
            #         total_loss_dict[key]  = value
                # total_loss_dict[key] += value
                
                # total_loss_dict['total_loss'] += 
        
        return total_loss_dict


    # predict
    def predict(self, model_path, x_test, batch_cls, keep_prob, phase):
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
                                                          self.batch_cls: batch_cls,
                                                          self.keep_prob: keep_prob, 
                                                          self.phase: phase})   # set phase to False for every prediction
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
