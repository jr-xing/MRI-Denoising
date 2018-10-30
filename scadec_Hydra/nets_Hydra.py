from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
import tensorflow.contrib as contrib

from scadec_Hydra import util
from scadec_Hydra.layers import *

        # "get_loss_dict": True,
        # "batch_size": 5,
        # "valid_size": 5,
        # 'n_classes':8
IFDEBUG = False
def dprint(string):
    global IFDEBUG
    if IFDEBUG:
        print(string)

def unet_decoder(x, keep_prob, phase, img_channels, truth_channels, layers=3, conv_times=3, features_root=16, filter_size=3, pool_size=2, summaries=True, get_loss_dict = True, batch_size = 5, valid_size = 5, structure={'type':'Nagini'}):#, n_classes = 8, structure='Hydra', neck_len = 3):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,img_channels]
    :param keep_prob: dropout probability tensor
    :param img_channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    :param structure: structure dict of network
    """
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    
    structure_type = structure.get('type', 'Nagini')
    n_classes = structure.get('n_classes', 16)    
    neck_len = structure.get('neck_len', 3)
    Ouroboros = structure.get('Ouroboros', True)
    
    with tf.variable_scope('geneator'):
        # Placeholder for the input image
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1,nx,ny,img_channels]))
        batch_size = tf.shape(x_image)[0]

        pools = OrderedDict()  # pooling layers
        deconvs = OrderedDict()  # deconvolution layer
        dw_h_convs = OrderedDict()  # down-side convs
        up_h_convs = OrderedDict()  # up-side convs
        # necks = OrderedDict()   # Necks for HydraNet- Xing
        necks = []
        # with tf.variable_scope('generator'):
        # conv the input image to desired feature maps
        in_node = conv2d_bn_relu(x_image, filter_size, features_root, keep_prob, phase, 'conv2feature_roots')

        # Down layers

        for layer in range(0, layers):
            features = 2**layer*features_root
            with tf.variable_scope('down_layer_' + str(layer)):
                for conv_iter in range(0, conv_times):
                    scope = 'conv_bn_relu_{}'.format(conv_iter)
                    conv = conv2d_bn_relu(in_node, filter_size, features, keep_prob, phase, scope)    
                    in_node = conv

                # store the intermediate result per layer
                dw_h_convs[layer] = in_node
                
                # down sampling
                if layer < layers-1:
                    with tf.variable_scope('pooling'):
                        pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                        in_node = pools[layer]
            
        in_node = dw_h_convs[layers-1]
            
        # Up layers
        if structure_type == 'HydraEr':
            for layer in range(layers-2, -1, -1):
                dprint('uplayer {}'.format(layer))
                features = 2**(layer+1)*features_root
                with tf.variable_scope('up_layer_' + str(layer)):
                    with tf.variable_scope('unsample_concat_layer'):
                        # number of features = lower layer's number of features
                        dprint('deconv...')
                        h_deconv = deconv2d_bn_relu(in_node, filter_size, features//2, pool_size, keep_prob, phase, 'unsample_layer')
                        h_deconv_concat = concat(dw_h_convs[layer], h_deconv)
                        deconvs[layer] = h_deconv_concat
                        in_node = h_deconv_concat
                    if layer > 0:
                        for conv_iter in range(0, conv_times):
                            scope = 'conv_bn_relu_{}'.format(conv_iter)
                            conv = conv2d_bn_relu(in_node, filter_size, features//2, keep_prob, phase, scope)    
                            in_node = conv         
                        up_h_convs[layer] = in_node
                    elif layer == 0:
                        dprint('total cls: {}'.format(n_classes))
                        for neck_idx in range(n_classes):
                            dprint('Processing neck {}'.format(neck_idx))
                            in_node = h_deconv_concat
                            for conv_iter in range(0, conv_times):
                                scope = 'conv_bn_relu_{}_cls_{}'.format(conv_iter, neck_idx)
                                conv = conv2d_bn_relu(in_node, filter_size, features//2, keep_prob, phase, scope)    
                                in_node = conv
                            necks.append(in_node)
                            up_h_convs['neck_{}'.format(neck_idx)] = in_node
            dprint('In total {} necks'.format(len(necks)))
            dprint(in_node.shape)
                    
        elif structure_type == 'Nagini' or structure_type == 'Hydra':
            for layer in range(layers-2, -1, -1):
                features = 2**(layer+1)*features_root
                with tf.variable_scope('up_layer_' + str(layer)):
                    with tf.variable_scope('unsample_concat_layer'):
                        # number of features = lower layer's number of features
                        h_deconv = deconv2d_bn_relu(in_node, filter_size, features//2, pool_size, keep_prob, phase, 'unsample_layer')
                        h_deconv_concat = concat(dw_h_convs[layer], h_deconv)
                        deconvs[layer] = h_deconv_concat
                        in_node = h_deconv_concat

                    for conv_iter in range(0, conv_times):
                        scope = 'conv_bn_relu_{}'.format(conv_iter)
                        conv = conv2d_bn_relu(in_node, filter_size, features//2, keep_prob, phase, scope)    
                        in_node = conv            

                    up_h_convs[layer] = in_node

            in_node = up_h_convs[0]
        else:
            raise ValueError('Unknown Net structure_type: {}'.format(structure_type))

        if structure_type == 'Nagini':
            # Output with residual
            with tf.variable_scope("conv2d_1by1"):
                if Ouroboros:
                    with tf.variable_scope('input_concat_layer'):
                        # number of features = lower layer's number of features
                        # h_deconv = deconv2d_bn_relu(in_node, filter_size, features//2, pool_size, keep_prob, phase, 'unsample_layer')
                        in_node = concat(x_image, in_node)                    
                output = conv2d(in_node, 1, truth_channels, keep_prob, 'conv2truth_channels')
                up_h_convs["out"] = output
        elif structure_type == 'HydraEr':
            for neck_idx in range(n_classes):
                with tf.variable_scope("conv2d_1by1_cls_{}".format(neck_idx)):
                    dprint('Processing neck {}'.format(neck_idx))
                    necks[neck_idx] = conv2d(necks[neck_idx], 1, truth_channels, keep_prob, 'conv2truth_channels')
                    up_h_convs["out"] = necks[neck_idx]

        elif structure_type == 'Hydra':
            # Necks - Xing        
            with tf.variable_scope("necks"):
                neck_features = [3,5,7]
                for neck_idx in range(n_classes):                
                    for neck_layer_idx in range(neck_len-1):
                        in_node = conv2d(in_node, 3, neck_features[neck_len-2 -neck_layer_idx], keep_prob, '{}_{}'.format(neck_idx, neck_layer_idx))
                    if Ouroboros:
                        with tf.variable_scope('input_concat_layer'):
                            # number of features = lower layer's number of features
                            # h_deconv = deconv2d_bn_relu(in_node, filter_size, features//2, pool_size, keep_prob, phase, 'unsample_layer')
                            in_node = concat(x_image, in_node)
                    output = conv2d(in_node, 1, truth_channels, keep_prob, '{}_conv2truth_channels'.format(neck_idx))
                    necks.append(output)
                        
                    # print('Shape of in_node')
                    # print(in_node.shape)
                    # Comment if OOM
                    # neck_1 = conv2d_bn_relu(in_node, 3, 5, keep_prob, phase, '{}_1'.format(neck_idx))
                    # print('Shape of neck1')
                    # print(neck_1.shape)
                    # neck_2 = conv2d_bn_relu(neck_1, 3, truth_channels, keep_prob, phase, '{}_2'.format(neck_idx))
                    # dprint('Shape of neck2')
                    # dprint(neck_2.shape)
                    # neck_3 = conv2d(neck_2, 1, truth_channels, keep_prob, '{}_conv2truth_channels'.format(neck_idx))
                    # neck_3 = conv2d(in_node, 1, truth_channels, keep_prob, '{}_conv2truth_channels'.format(neck_idx))
                    # dprint('Shape of neck3')
                    # dprint(neck_3.shape)
                    # necks[neck_idx] = neck_3
                    # necks.append(neck_3)
        else:
            raise ValueError('Unknown Net structure_type: {}'.format(structure_type))
        
        if summaries:
            # for i, (c1, c2) in enumerate(convs):
            #     tf.summary.image('summary_conv_%02d_01'%i, get_image_summary(c1))
            #     tf.summary.image('summary_conv_%02d_02'%i, get_image_summary(c2))
                
            for k in pools.keys():
                tf.summary.image('summary_pool_%02d'%k, get_image_summary(pools[k]))
            
            for k in deconvs.keys():
                tf.summary.image('summary_deconv_concat_%02d'%k, get_image_summary(deconvs[k]))
                
            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k])

        # Xing
        # return output#, dw_h_convs, up_h_convs
        if structure_type == 'Nagini':
            return output
        elif structure_type == 'Hydra' or structure_type == 'HydraEr':
            return necks
        else:
            raise ValueError('Unknown Net structure_type: {}'.format(structure_type))

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V


def unet_decoder_noGAN(x, keep_prob, phase, img_channels, truth_channels, layers=3, conv_times=3, features_root=16, filter_size=3, pool_size=2, summaries=True, get_loss_dict = True, batch_size = 5, valid_size = 5, structure={'type':'Nagini'}):#, n_classes = 8, structure='Hydra', neck_len = 3):
    print('------------------------')
    print('USING unet_decoder_noGAN')
    print('------------------------')

    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,img_channels]
    :param keep_prob: dropout probability tensor
    :param img_channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    structure_type = structure.get('type', 'Nagini')
    n_classes = structure.get('n_classes', 16)    
    neck_len = structure.get('neck_len', 3)
    Ouroboros = structure.get('Ouroboros', True)
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))

    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,img_channels]))
    batch_size = tf.shape(x_image)[0]

    pools = OrderedDict()  # pooling layers
    deconvs = OrderedDict()  # deconvolution layer
    dw_h_convs = OrderedDict()  # down-side convs
    up_h_convs = OrderedDict()  # up-side convs
    # necks = OrderedDict()   # Necks for HydraNet- Xing
    necks = []

    # conv the input image to desired feature maps
    in_node = conv2d_bn_relu(x_image, filter_size, features_root, keep_prob, phase, 'conv2feature_roots')

    # Down layers

    for layer in range(0, layers):
        features = 2**layer*features_root
        with tf.variable_scope('down_layer_' + str(layer)):
            for conv_iter in range(0, conv_times):
                scope = 'conv_bn_relu_{}'.format(conv_iter)
                conv = conv2d_bn_relu(in_node, filter_size, features, keep_prob, phase, scope)    
                in_node = conv

            # store the intermediate result per layer
            dw_h_convs[layer] = in_node
            
            # down sampling
            if layer < layers-1:
                with tf.variable_scope('pooling'):
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    in_node = pools[layer]
        
    in_node = dw_h_convs[layers-1]
        
    # Up layers
    if structure_type == 'HydraEr':
        for layer in range(layers-2, -1, -1):
            dprint('uplayer {}'.format(layer))
            features = 2**(layer+1)*features_root
            with tf.variable_scope('up_layer_' + str(layer)):
                with tf.variable_scope('unsample_concat_layer'):
                    # number of features = lower layer's number of features
                    dprint('deconv...')
                    h_deconv = deconv2d_bn_relu(in_node, filter_size, features//2, pool_size, keep_prob, phase, 'unsample_layer')
                    h_deconv_concat = concat(dw_h_convs[layer], h_deconv)
                    deconvs[layer] = h_deconv_concat
                    in_node = h_deconv_concat
                if layer > 0:
                    for conv_iter in range(0, conv_times):
                        scope = 'conv_bn_relu_{}'.format(conv_iter)
                        conv = conv2d_bn_relu(in_node, filter_size, features//2, keep_prob, phase, scope)    
                        in_node = conv         
                    up_h_convs[layer] = in_node
                elif layer == 0:
                    dprint('total cls: {}'.format(n_classes))
                    for neck_idx in range(n_classes):
                        dprint('Processing neck {}'.format(neck_idx))
                        in_node = h_deconv_concat
                        for conv_iter in range(0, conv_times):
                            scope = 'conv_bn_relu_{}_cls_{}'.format(conv_iter, neck_idx)
                            conv = conv2d_bn_relu(in_node, filter_size, features//2, keep_prob, phase, scope)    
                            in_node = conv
                        necks.append(in_node)
                        up_h_convs['neck_{}'.format(neck_idx)] = in_node
        dprint('In total {} necks'.format(len(necks)))
        dprint(in_node.shape)
                
    elif structure_type == 'Nagini' or structure_type == 'Hydra':
        for layer in range(layers-2, -1, -1):
            features = 2**(layer+1)*features_root
            with tf.variable_scope('up_layer_' + str(layer)):
                with tf.variable_scope('unsample_concat_layer'):
                    # number of features = lower layer's number of features
                    h_deconv = deconv2d_bn_relu(in_node, filter_size, features//2, pool_size, keep_prob, phase, 'unsample_layer')
                    h_deconv_concat = concat(dw_h_convs[layer], h_deconv)
                    deconvs[layer] = h_deconv_concat
                    in_node = h_deconv_concat

                for conv_iter in range(0, conv_times):
                    scope = 'conv_bn_relu_{}'.format(conv_iter)
                    conv = conv2d_bn_relu(in_node, filter_size, features//2, keep_prob, phase, scope)    
                    in_node = conv            

                up_h_convs[layer] = in_node

        in_node = up_h_convs[0]
    else:
        raise ValueError('Unknown Net structure_type: {}'.format(structure_type))

    if structure_type == 'Nagini':
        # Output with residual
        with tf.variable_scope("conv2d_1by1"):
            output = conv2d(in_node, 1, truth_channels, keep_prob, 'conv2truth_channels')
            up_h_convs["out"] = output
    elif structure_type == 'HydraEr':
        for neck_idx in range(n_classes):
            with tf.variable_scope("conv2d_1by1_cls_{}".format(neck_idx)):
                dprint('Processing neck {}'.format(neck_idx))
                necks[neck_idx] = conv2d(necks[neck_idx], 1, truth_channels, keep_prob, 'conv2truth_channels')
                up_h_convs["out"] = necks[neck_idx]

    elif structure_type == 'Hydra':
        # Necks - Xing        
        with tf.variable_scope("necks_"):
            neck_features = [3,5,7]
            for neck_idx in range(n_classes):                
                for neck_layer_idx in range(neck_len-1):
                    in_node = conv2d(in_node, 3, neck_features[neck_len-2 -neck_layer_idx], keep_prob, '{}_{}'.format(neck_idx, neck_layer_idx))
                output = conv2d(in_node, 1, truth_channels, keep_prob, '{}_conv2truth_channels'.format(neck_idx))
                necks.append(output)
                    
                # print('Shape of in_node')
                # print(in_node.shape)
                # Comment if OOM
                # neck_1 = conv2d_bn_relu(in_node, 3, 5, keep_prob, phase, '{}_1'.format(neck_idx))
                # print('Shape of neck1')
                # print(neck_1.shape)
                # neck_2 = conv2d_bn_relu(neck_1, 3, truth_channels, keep_prob, phase, '{}_2'.format(neck_idx))
                # dprint('Shape of neck2')
                # dprint(neck_2.shape)
                # neck_3 = conv2d(neck_2, 1, truth_channels, keep_prob, '{}_conv2truth_channels'.format(neck_idx))
                # neck_3 = conv2d(in_node, 1, truth_channels, keep_prob, '{}_conv2truth_channels'.format(neck_idx))
                # dprint('Shape of neck3')
                # dprint(neck_3.shape)
                # necks[neck_idx] = neck_3
                # necks.append(neck_3)
    else:
        raise ValueError('Unknown Net structure_type: {}'.format(structure_type))
    
    if summaries:
        # for i, (c1, c2) in enumerate(convs):
        #     tf.summary.image('summary_conv_%02d_01'%i, get_image_summary(c1))
        #     tf.summary.image('summary_conv_%02d_02'%i, get_image_summary(c2))
            
        for k in pools.keys():
            tf.summary.image('summary_pool_%02d'%k, get_image_summary(pools[k]))
        
        for k in deconvs.keys():
            tf.summary.image('summary_deconv_concat_%02d'%k, get_image_summary(deconvs[k]))
            
        for k in dw_h_convs.keys():
            tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k])

        for k in up_h_convs.keys():
            tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k])

    # Xing
    # return output#, dw_h_convs, up_h_convs
    if structure_type == 'Nagini':
        return output
    elif structure_type == 'Hydra' or structure_type == 'HydraEr':
        return necks
    else:
        raise ValueError('Unknown Net structure_type: {}'.format(structure_type))