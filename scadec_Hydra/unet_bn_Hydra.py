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

from scadec_Hydra import util
from scadec_Hydra.layers import *
from scadec_Hydra.nets_Hydra import *

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./"))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./scadec/"))


# from IPython.core.debugger import Tracer
# import ipdb
IFDEBUG = False
# def dprint(string):
#     global IFDEBUG
#     if IFDEBUG:
#         print(string)

class Unet_bn(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    :param kwargs: args passed to create_net function. 
    """
   
    def __init__(self, img_channels=3, truth_channels=3, cost_dict_list=[{'name':'l2'}], x_shape = None, y_shape = None, **kwargs):
        tf.reset_default_graph()

        # basic variables
        self.summaries = kwargs.get("summaries", True)
        self.img_channels = img_channels
        self.truth_channels = truth_channels
        self.batch_size = kwargs.get("batch_size", 5)
        self.ifGAN = kwargs.pop('GAN', False)

        # placeholders for input x and y
        if x_shape != None:
            # Assign shape to use VGGNet - Xing
            self.x = tf.placeholder("float", shape=x_shape)
            self.y = tf.placeholder("float", shape=y_shape)
        else:
            self.x = tf.placeholder("float", shape=[None, None, None, img_channels])
            self.y = tf.placeholder("float", shape=[None, None, None, truth_channels])
            # Added by Xing
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)
        self.current_epoch = tf.placeholder(tf.int32, name='current_epoch')
        self.total_epochs = tf.placeholder(tf.int32, name='total_epochs')


        # reused variables
        # self.batch_size = tf.shape(self.x)[0]
        # self.batch_size = 5
        self.nx = tf.shape(self.x)[1]
        self.ny = tf.shape(self.x)[2]
        self.num_examples = tf.shape(self.x)[0]

        # variables need to be calculated
        # self.recons = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
        # Xing
        self.structure = kwargs.get('structure','Hydra')
        if not self.ifGAN:
            if self.structure == 'Hydra' or self.structure == 'HydraEr':
                self.necks = unet_decoder_noGAN(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
            elif self.structure == 'Nagini':
                self.recons = unet_decoder_noGAN(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
            else:
                raise ValueError('Unknown Net Structure: '+self.structure)        
        else:
            if self.structure == 'Hydra' or self.structure == 'HydraEr':
                self.necks = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
            elif self.structure == 'Nagini':
                self.recons = unet_decoder(self.x, self.keep_prob, self.phase, self.img_channels, self.truth_channels, **kwargs)
            else:
                raise ValueError('Unknown Net Structure: '+self.structure)        
        
        # Additional info
        # Array Version
        # self.batch_cls = tf.placeholder(tf.int32,[None], name='batch_cls')
        # one hot version
        self.batch_cls = tf.placeholder(tf.int32,[None, kwargs.get('n_classes',1)], name='batch_cls')

        # Xing
        self.loss_dict = self._get_cost(cost_dict_list)
        self.loss_no_disc = self.loss_dict['total_loss']
        self.valid_loss_dict = self._get_cost(cost_dict_list)

        # GAN
        if self.ifGAN:
            self.disc_loss, self.disc_loss_real = self._get_discriminator_loss()
        else:
            self.disc_loss = None
            self.disc_loss_real = None
        # Total loss
        
        self.loss = self.loss_dict['total_loss']
        self.valid_loss = self.valid_loss_dict['total_loss']
                
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
        
    def _get_cost(self, cost_dict_list):
        """
        Constructs the cost function.

        """
        #Tracer()        
        # ddprint('_get_cost')
        dprint('Computing Cost...')
        # total_pixels = self.nx * self.ny * self.truth_channels

        def get_mask(mode = 'default', h = 320, w = 320):
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
                scaleX = 1.2
                scaleY = 0.8
                normX = np.linspace(norm.ppf(0.001, scale=scaleX),norm.ppf(0.999,scale=scaleX), w)
                normY = np.linspace(norm.ppf(0.001, scale=scaleY),norm.ppf(0.999,scale=scaleY), h)
                normXX, normYY = np.meshgrid(normX, normY)
                maskN = norm.pdf(np.abs(normXX)+np.abs(normYY))
                maskN = (maskN-np.min(maskN))/np.max(maskN)
                return np.float32(np.reshape(maskN,[w,h,1]))        

        def non_max_suppression(input, window_size = 3):
            # From: https://stackoverflow.com/questions/42879109/tensorflow-non-maximum-suppression
            # input: B x W x H x C
            pooled = tf.nn.max_pool(input, ksize=[1, window_size, window_size, 1], strides=[1,1,1,1], padding='SAME')
            output = tf.where(tf.equal(input, pooled), input, tf.zeros_like(input))

            # Note: if input has negative values, the suppressed values can be higher than original 
            return output # output: B X W X H x C

        def get_edge(img, operator, get_XY = False, NMS = False, NMS_window_size = 3):
            # https://blog.csdn.net/huanghuangjin/article/details/81130171
            # https://www.cnblogs.com/wxl845235800/p/7700867.html
            if operator == 'gradient':
                gradX = tf.reshape(tf.constant([[0,0,0],[1,0,-1],[0,0,0]],tf.float32),[3, 3, 1, 1])
                gradY = tf.reshape(tf.constant([[0,1,0],[0,0,0],[0,-1,0]],tf.float32),[3, 3, 1, 1])
                imgX = tf.nn.conv2d(img, gradX, strides=[1,1,1,1], padding='SAME')
                imgY = tf.nn.conv2d(img, gradY, strides=[1,1,1,1], padding='SAME')
                if NMS:
                    imgX = non_max_suppression(imgX, NMS_window_size)
                    imgY = non_max_suppression(imgY, NMS_window_size)

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

        
        def get_losses(x, y, cost_dict_list):
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
                
                else:
                    raise ValueError("Unknown cost function: "+cost_dict['name'])

                if cost_dict.get('upper_bound',False):
                    #current_loss =  tf.clip_by_value(current_loss, 0.0, cost_dict.get('upper_bound',0.5))
                    current_loss =  tf.clip_by_norm(current_loss, cost_dict.get('upper_bound',0.5))
                
                
                loss += cost_dict['weight']*current_loss
                loss_dict[current_loss_name] = current_loss
                #loss +=  tf.cond(check_if_valid(cost_dict), lambda: cost_dict['weight']*current_loss, lambda: 0.0)                
                #loss_dict[current_loss_name] = tf.cond(check_if_valid(cost_dict), lambda: current_loss, lambda: 0.0)

            loss_dict['total_loss'] = loss
            return loss_dict
        
        # def head(img):
        #     # global sliceIdx
        #     return img
        
        # Hydra
        # batch_size = tf.shape(self.x)[0]
        # Create total loss dict
        total_loss_dict = {}
        total_loss_dict['total_loss'] = 0
        for cost_dict in cost_dict_list:
            if cost_dict['name'] == 'edge':
                total_loss_dict[cost_dict['edge_type']]=0
            else:
                total_loss_dict[cost_dict['name']] = 0
        
        dprint('Structure: '+self.structure)
        if self.structure == 'Hydra' or self.structure == 'HydraEr':
            dprint('Init self.recons:')
            self.recons = None
            dprint(self.recons)
            dprint('batch_size: {}'.format(self.batch_size))
            for img_idx in range(self.batch_size):            
                img_class_onehot = self.batch_cls[img_idx, :]
                recon = tf.reshape(tf.dynamic_partition(self.necks, img_class_onehot, 2, name = 'part_necks')[0][0][img_idx,:,:,:],[1,320,320,self.truth_channels], name = 'reshape_recon')                  
                y = tf.reshape(self.y[img_idx,:,:,:],[1,320,320,self.truth_channels])
                
                loss_dict = get_losses(recon, y, cost_dict_list)                
                
                if self.recons == None:
                    dprint('Create self.recons')
                    self.recons = recon
                else:
                    dprint('Concat self.recons')
                    self.recons = tf.concat([self.recons, recon],axis=0)                                        
                    

                for key, value in loss_dict.items():
                    total_loss_dict[key] += value

        elif self.structure == 'Nagini':
            loss_dict = get_losses(self.recons, self.y, cost_dict_list)

            for key, value in loss_dict.items():
                total_loss_dict[key] += value

        return total_loss_dict



        
        # loss = 0
        # loss_dict = {}
        # for cost_dict in cost_dict_list:
        #     if cost_dict['name'] == 'l2' or cost_dict['name'] == 'mean_square_error':                
        #         mask = get_mask(mode=cost_dict.get('mask',None),img_h=320,img_w=320)                
        #         loss_l2 = tf.losses.mean_squared_error(tf.multiply(self.recons,mask), tf.multiply(self.y,mask))
        #         current_loss = loss_l2
        #         current_loss_name = 'mean_square_error'
        #         #loss += cost_dict['weight']*loss_l2
        #         #loss_dict['meaｎ_square_loss'] = loss_l2

        #     elif cost_dict['name'] == 'edge':
        #         mask = get_mask(mode=cost_dict.get('mask',None),img_h=320,img_w=320)
        #         if cost_dict.get('mask_before_operate',False):
        #             # Get masked images
        #             recons_masked = tf.multiply(self.recons, mask)
        #             y_masked = tf.multiply(self.y, mask)

        #             # Compute edge loss
        #             if cost_dict.get('get_XY',False):
        #                 edge_recons_X,edge_recons_Y = get_edge(recons_masked, operator=cost_dict['edge_type'], get_XY=True)
        #                 edge_y_X,edge_y_Y = get_edge(y_masked, operator=cost_dict['edge_type'], get_XY=True)
        #                 loss_masked_edge = tf.losses.absolute_difference(edge_recons_X, edge_y_X)+tf.losses.absolute_difference(edge_recons_Y, edge_y_Y)
        #             else:
        #                 edge_recons = get_edge(recons_masked, operator=cost_dict['edge_type'])
        #                 edge_y = get_edge(y_masked, operator=cost_dict['edge_type'])
        #                 loss_masked_edge = tf.losses.mean_squared_error(edge_recons, edge_y)   
        #             # Loss
        #             current_loss = loss_masked_edge
        #             current_loss_name = cost_dict['edge_type']

        #         else:                    
        #             if cost_dict.get('get_XY',False):
        #                 edge_recons_X,edge_recons_Y = get_edge(self.recons, operator=cost_dict['edge_type'], get_XY=True)
        #                 edge_y_X,edge_y_Y = get_edge(self.y, operator=cost_dict['edge_type'], get_XY=True)
        #                 loss_edge_masked = tf.losses.absolute_difference(tf.multiply(edge_recons_X,mask), tf.multiply(edge_y_X,mask)) + \
        #                                     tf.losses.absolute_difference(tf.multiply(edge_recons_Y,mask), tf.multiply(edge_y_Y,mask))
        #             else:
        #                 edge_recons = get_edge(self.recons, operator=cost_dict['edge_type'])
        #                 edge_y = get_edge(self.y, operator=cost_dict['edge_type'])
        #                 loss_edge_masked = tf.losses.mean_squared_error(tf.multiply(edge_recons, mask), tf.multiply(edge_y, mask))
        #             current_loss = loss_edge_masked
        #             current_loss_name = cost_dict['edge_type']

        #     else:
        #         raise ValueError("Unknown cost function: "+cost_dict['name'])

        #     if cost_dict.get('upper_bound',False):
        #         #current_loss =  tf.clip_by_value(current_loss, 0.0, cost_dict.get('upper_bound',0.5))
        #         current_loss =  tf.clip_by_norm(current_loss, cost_dict.get('upper_bound',0.5))
            
        #     loss += cost_dict['weight']*current_loss
        #     loss_dict[current_loss_name] = current_loss

        # loss_dict['total_loss'] = loss
        # return loss_dict
    
    def _get_discriminator_loss(self):
        # CP from pix2pix
        EPS = 1e-12
        def discrim_conv(batch_input, out_channels, stride):
            padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        
        def lrelu(x, a):
            with tf.name_scope("lrelu"):
                # adding these together creates the leak part and linear part
                # then cancels them out by subtracting/adding an absolute value term
                # leak: a*x/2 - a*abs(x)/2
                # linear: x/2 + abs(x)/2

                # this block looks like it has 2 inputs on the graph unless we do this
                x = tf.identity(x)
                return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
                
        def batchnorm(inputs):
            return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

        def create_discriminator(discrim_inputs, discrim_targets):            
            n_layers = 2
            ndf = 32    # number of discriminator filters in first conv layer
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([discrim_inputs, discrim_targets], axis=3)

            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope("layer_1"):
                convolved = discrim_conv(input, ndf, stride=2)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                    normalized = batchnorm(convolved)
                    rectified = lrelu(normalized, 0.2)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = discrim_conv(rectified, out_channels=1, stride=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]
        
        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        # Discriminator take (condition, image(fake/real))
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = create_discriminator(self.x, self.y)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = create_discriminator(self.x, self.recons)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0        
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
            discrim_loss_real = tf.reduce_mean(-(tf.log(predict_real + EPS)))
        
        weight = 1e-4
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        
        self.loss_dict['GAN_generator'] = gen_loss_GAN
        self.loss_dict['total_loss'] += gen_loss_GAN * weight
        
        self.valid_loss_dict['GAN_generator'] = gen_loss_GAN
        self.valid_loss_dict['total_loss'] += gen_loss_GAN * weight
        
        return discrim_loss, discrim_loss_real



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
