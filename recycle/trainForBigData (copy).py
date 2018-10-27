from scadec.unet_bn import Unet_bn
from scadec.train import Trainer_bn

from scadec import image_util

import scipy.io as spio
import numpy as np
import os
import h5py


####################################################
####                 FUNCTIONS                   ###
####################################################

# make the data a 4D vector
def preprocess(data, channels):
	nx = data.shape[1]
	ny = data.shape[2]
	return data.reshape((-1, nx, ny, channels))

def h5py_mat2npy(datemat):
    a = h5py.File(datemat)
    # test=a[a.keys()[i]]
    #test=a['train_nbm_5']

    for data in a:
        test=np.array(a[data])
        test=test.T
        # Chang here for croped data - Xing
        if len(np.shape(test)) == 3:
            nx,ny = np.shape(test)[1:]
            chs = 1
        elif len(np.shape(test)) == 4:
            nx,ny, chs = np.shape(test)[1:]
        #nx = 120
        #ny = 120
        test_x = np.reshape(test,[-1,nx,ny,chs])
    
    return test_x

####################################################
####              HYPER-PARAMETERS               ###
####################################################

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind; # 0,1,2,3


# Because we have real & imaginary part of our input, data_channels is set to 2

####################################################
####                DATA LOADING                 ###
####################################################

"""
	here loads all the data we need for training and validating.

"""

# def gray2rgb(gray_img, mode = 'copy'):
#     if mode == 'copy':
#         # https://www.zhihu.com/question/266789638
#         return np.concatenate((gray_img, gray_img, gray_img), -1)
#     elif mode == 'peusodo color':
#         # https://stackoverflow.com/questions/10901085/range-values-to-pseudocolor
#         pass
#     elif mode == 'seq_frame':
#         # gray_img = vdata        
#         imgs = np.zeros(shape = gray_img.shape[:3]+(3,))
#         # Previous Slice Channel
#         imgs[0:,:,:,0] = np.squeeze(gray_img[0,:,:,:])
#         imgs[1:,:,:,0] = np.squeeze(gray_img[:-1,:,:,:])
#         # Current Slice Channel
#         imgs[:,:,:,1] = np.squeeze(gray_img)
#         # Next Slice Channel
#         imgs[:-1,:,:,2] = np.squeeze(gray_img[1:,:,:,:])
#         imgs[-1,:,:,2] = np.squeeze(gray_img[-1,:,:,:])
#         return imgs
#     elif mode == 'seq_frame_5':
#         # gray_img = vdata        
#         imgs = np.zeros(shape = gray_img.shape[:3]+(5,))
#         # Previous Slices Channel
#         # Prev -2 chennel
#         imgs[0,:,:,0] = np.squeeze(gray_img[0,:,:,:])
#         imgs[1,:,:,0] = np.squeeze(gray_img[0,:,:,:])
#         imgs[2:,:,:,0] = np.squeeze(gray_img[:-2:,:,:])
#         # Prev -1 chennel
#         imgs[0,:,:,1] = np.squeeze(gray_img[0,:,:,:])
#         imgs[1:,:,:,1] = np.squeeze(gray_img[:-1,:,:,:])
#         # Current Slice Channel
#         imgs[:,:,:,2] = np.squeeze(gray_img)
#         # Post Slices Channel
#         # Post 1 chennel        
#         imgs[:-1,:,:,3] = np.squeeze(gray_img[1:,:,:,:])
#         imgs[-1,:,:,3] = np.squeeze(gray_img[-1,:,:,:])
#         # Post 2 chennel
#         imgs[:-2,:,:,4] = np.squeeze(gray_img[2:,:,:,:])
#         imgs[-2,:,:,4] = np.squeeze(gray_img[-1,:,:,:])
#         imgs[-1,:,:,4] = np.squeeze(gray_img[-1,:,:,:])
#         return imgs

# 'l2_3C_motion_reg_l1_mid_drop_0.75'
# parameter_str = 'l2_3C_motion_masked_gradient_masked_reg_no_drop_0.9_FULL_SEG'
# l2_3C_motion_masked_masked_Sobel_mid5_reg_no_drop_0.9_FULL_SEG_w2
# parameter_str = 'l2_3C_motion_masked_masked_Sobel_reg_no_drop_0.9_FULL_SEG'
parameter_str = 'l2_3C_motion_masked_masked_Sobel_reg_no_drop_0.9_FULL_SEG_w2'

print('Loading Data...')
if parameter_str == 'l2_3C_motion_reg_no_drop_0.75':
    loss = 'mean_squared_error'
    data_channels = 3 # xiaojian
    truth_channels = 1
    
    data = h5py_mat2npy('train_np/traOb_neigh_motion.mat')
    truths = h5py_mat2npy('train_np/traGt.mat')
    data_provider = image_util.SimpleDataProvider(data, truths)
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_3C_motion_reg_no_drop_0.75_FULL_data':
    loss = 'mean_squared_error'
    data_channels = 3 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_2.mat')
    data  = np.concatenate([data1, data2], axis=0)
    del(data1, data2)

    truths1 = h5py_mat2npy('train_np/traGt_FULL_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_part_2.mat')
    truths  = np.concatenate([truths1, truths2], axis=0)
    del(truths1, truths2)

    data_provider = image_util.SimpleDataProvider(data, truths)        
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_3C_motion_gradient_reg_no_drop_0.75_FULL_data':
    loss = 'mean_squared_error_gradient'
    data_channels = 3 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_2.mat')
    data  = np.concatenate([data1, data2], axis=0)
    del(data1, data2)

    truths1 = h5py_mat2npy('train_np/traGt_FULL_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_part_2.mat')
    truths  = np.concatenate([truths1, truths2], axis=0)
    del(truths1, truths2)

    data_provider = image_util.SimpleDataProvider(data, truths)        
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)  

elif parameter_str == 'l2_3C_motion_masked_gradient_masked_reg_no_drop_0.75_FULL_data':
    loss = 'mean_squared_error_masked_gradient_masked'
    data_channels = 3 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_2.mat')
    data  = np.concatenate([data1, data2], axis=0)
    del(data1, data2)

    truths1 = h5py_mat2npy('train_np/traGt_FULL_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_part_2.mat')
    truths  = np.concatenate([truths1, truths2], axis=0)
    del(truths1, truths2)

    data_provider = image_util.SimpleDataProvider(data, truths)        
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_3C_motion_masked_gradient_masked_reg_no_drop_0.75_FULL_data_w2':
    loss = 'mean_squared_error_masked_gradient_masked'
    cost_kwargs = {
        "mask_type": "default",
        "grad_weight": 50
        }
    kwargs = {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
    data_channels = 3 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_2.mat')
    data  = np.concatenate([data1, data2], axis=0)
    print(np.shape(data))
    del(data1, data2)

    truths1 = h5py_mat2npy('train_np/traGt_FULL_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_part_2.mat')
    truths  = np.concatenate([truths1, truths2], axis=0)
    del(truths1, truths2)
    print(np.shape(truths))

    data_provider = image_util.SimpleDataProvider(data, truths)        
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_3C_motion_masked_gradient_masked_mid5_reg_no_drop_0.75_FULL_data_w2':
    loss = 'mean_squared_error_masked_gradient_masked'
    cost_kwargs = {
        "mask_type": "mid5",
        "grad_weight": 50
        }
    kwargs = {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
    dropout_rate = 0.75
    data_channels = 3 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_2.mat')
    data  = np.concatenate([data1, data2], axis=0)
    print(np.shape(data))
    del(data1, data2)

    truths1 = h5py_mat2npy('train_np/traGt_FULL_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_part_2.mat')
    truths  = np.concatenate([truths1, truths2], axis=0)
    del(truths1, truths2)
    print(np.shape(truths))

    data_provider = image_util.SimpleDataProvider(data, truths)        
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_3C_motion_masked_masked_Sobel_reg_no_drop_0.9_FULL_SEG_w2':
    # Date: Oct 5
    loss = 'mean_squared_error_masked_masked_sobel'
    cost_kwargs = {
        "mask_type": "norm",
        "Sobel_weight": 50
        }
    kwargs = {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
    dropout_rate = 9
    data_channels = 3 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
    data3 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_3.mat')
    data  = np.concatenate([data1, data2, data3], axis=0)    
    del(data1, data2, data3)
    #print(np.shape(data))

    truths1 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_2.mat')
    truths3 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_3.mat')
    truths  = np.concatenate([truths1, truths2, truths3], axis=0)
    del(truths1, truths2, truths3)

    data_provider = image_util.SimpleDataProvider(data, truths)        
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_3C_motion_masked_gradient_masked_reg_no_drop_0.75_FULL_SEG':
    loss = 'mean_squared_error_masked_gradient_masked'
    cost_kwargs = {
        "mask_type": "default",
        "grad_weight": 10
        }
    kwargs = {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
    data_channels = 3 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
    data3 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_3.mat')
    data  = np.concatenate([data1, data2, data3], axis=0)    
    del(data1, data2, data3)
    #print(np.shape(data))

    truths1 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_2.mat')
    truths3 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_3.mat')
    truths  = np.concatenate([truths1, truths2, truths3], axis=0)
    del(truths1, truths2, truths3)
    #print(np.shape(truths))

    data_provider = image_util.SimpleDataProvider(data, truths)        
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)


elif parameter_str == 'l2_3C_motion_masked_gradient_masked_reg_no_drop_0.9_FULL_SEG':
    loss = 'mean_squared_error_masked_gradient_masked'
    cost_kwargs = {
        "mask_type": "default",
        "grad_weight": 10
        }
    kwargs = {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
    dropout_rate = 0.9
    data_channels = 3 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
    data3 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_3.mat')
    data  = np.concatenate([data1, data2, data3], axis=0)    
    del(data1, data2, data3)
    #print(np.shape(data))

    truths1 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_2.mat')
    truths3 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_3.mat')
    truths  = np.concatenate([truths1, truths2, truths3], axis=0)
    del(truths1, truths2, truths3)
    #print(np.shape(truths))

    data_provider = image_util.SimpleDataProvider(data, truths)        
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_3C_motion_masked_masked_Sobel_reg_no_drop_0.9_FULL_SEG':
    loss = 'mean_squared_error_masked_masked_sobel'
    cost_kwargs = {
        "mask_type": "norm",
        "Sobel_weight": 15
        }
    kwargs = {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
    dropout_rate = 0.9
    data_channels = 3 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
    data3 = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_3.mat')
    data  = np.concatenate([data1, data2, data3], axis=0)    
    del(data1, data2, data3)
    #print(np.shape(data))

    truths1 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_2.mat')
    truths3 = h5py_mat2npy('train_np/traGt_FULL_SEG_part_3.mat')
    truths  = np.concatenate([truths1, truths2, truths3], axis=0)
    del(truths1, truths2, truths3)
    #print(np.shape(truths))

    data_provider = image_util.SimpleDataProvider(data, truths)        
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_3C_motion_masked_reg_no_drop_0.75':
    loss = 'mean_squared_error_mask'
    data_channels = 3 # xiaojian
    truth_channels = 1
    
    data = h5py_mat2npy('train_np/traOb_neigh_motion.mat')
    truths = h5py_mat2npy('train_np/traGt.mat')
    data_provider = image_util.SimpleDataProvider(data, truths)
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_3C_motion_masked_reg_no_drop_0.75_FULL_data':
    loss = 'mean_squared_error_mask'
    data_channels = 3 # xiaojian
    truth_channels = 1
    
    data1 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_neigh_motion_part_2.mat')
    data  = np.concatenate([data1, data2], axis=0)
    del(data1, data2)

    truths1 = h5py_mat2npy('train_np/traGt_FULL_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_part_2.mat')
    truths  = np.concatenate([truths1, truths2], axis=0)
    del(truths1, truths2)

    data_provider = image_util.SimpleDataProvider(data, truths)
    
    vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)


elif parameter_str == 'l2_mask_reg_no_drop_0.75_FULL_data':
    loss = 'mean_squared_error_mask'

    data_channels = 1 # xiaojian
    truth_channels = 1

    data1 = h5py_mat2npy('train_np/traOb_FULL_part_1.mat')
    data2 = h5py_mat2npy('train_np/traOb_FULL_part_2.mat')
    data  = np.concatenate([data1, data2], axis=0)
    del(data1, data2)

    truths1 = h5py_mat2npy('train_np/traGt_FULL_part_1.mat')
    truths2 = h5py_mat2npy('train_np/traGt_FULL_part_2.mat')
    truths  = np.concatenate([truths1, truths2], axis=0)
    del(truths1, truths2)

    data_provider = image_util.SimpleDataProvider(data, truths)
    
    vdata = h5py_mat2npy('valid_np/valOb.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)

elif parameter_str == 'l2_reg_no_drop_0.75':
    loss = 'mean_squared_error'

    data_channels = 1 # xiaojian
    truth_channels = 1

    data = h5py_mat2npy('train_np/traOb.mat')
    truths = h5py_mat2npy('train_np/traGt.mat')
    data_provider = image_util.SimpleDataProvider(data, truths)
    
    vdata = h5py_mat2npy('valid_np/valOb.mat')
    vtruths = h5py_mat2npy('valid_np/valGt.mat')
    valid_provider = image_util.SimpleDataProvider(vdata, vtruths)
#-- Training Data --#
# data = data_mat['traOb']
# Why still (2880,160,160,1) - Xing
#data = preprocess(data_mat, data_channels)    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
#truths = preprocess(truths_mat, truth_channels)

# data_provider = image_util.SimpleDataProvider(data_3, truths)

#-- Validating Data --#

# Change to h5py to avoid error - Xing
# https://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python
# https://blog.csdn.net/qq_36427732/article/details/78044918

# vdata_mat = h5py.File('valid_np/valOb_Crop.mat','r')
# vtruths_mat = h5py.File('valid_np/valGt_Crop.mat', 'r')
#vdata_mat = spio.loadmat('valid_np/valOb_Crop.mat', squeeze_me=True)
#vtruths_mat = spio.loadmat('valid_np/valGt_Crop.mat', squeeze_me=True)

# vdata = vdata_mat['valOb']
# vtruths = vtruths_mat['valGt']
#vdata = preprocess(vdata_mat, data_channels)
#vtruths = preprocess(vtruths_mat, truth_channels)

# valid_provider = image_util.SimpleDataProvider(vdata_3, vtruths)


####################################################
####                  NETWORK                    ###
####################################################

"""
	here we specify the neural network.

"""

#-- Network Setup --#
# set up args for the unet

# I reject my humanity, JoJo! 
# net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost="l2_perceptual", cost_kwargs = cost_kwargs, x_shape = [5,160,160,3], y_shape = [5,160,160,3], **kwargs)
# net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost="perceptual", cost_kwargs = cost_kwargs, x_shape = [5,160,160,3], y_shape = [5,160,160,3], **kwargs)
# net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost="mean_squared_error", **kwargs)
net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost=loss, cost_kwargs=cost_kwargs, **kwargs)


####################################################
####                 TRAINING                    ###
####################################################

# args for training
batch_size = 5 # batch size for training
valid_size = 5  # batch size for validating
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

# # output paths for results
# parameter_str = 'l2_seq_5_reg_no_drop_0.75'
# parameter_str = 'l2_3C_motion_reg_l1_mid_drop_0.75'
# parameter_str = 'l2_perceptual_reg_no_drop_0.75'
# parameter_str = 'l2_perceptual_w_{}_f_{}_s_{}_tv_{}_reg_no_drop_0.75'.format(cost_kwargs['perceptual']['lambda_f'], cost_kwargs['perceptual']['lambda_s'] ,cost_kwargs['perceptual']['lambda_tv'])
# parameter_str =  'perceptual_f_{}_s_{}_tv_{}_reg_no_drop_0.75'.format(cost_kwargs['perceptual']['lambda_f'], cost_kwargs['perceptual']['lambda_s'] ,cost_kwargs['perceptual']['lambda_tv'])
output_path = 'gpu' + gpu_ind + '/' + parameter_str + '/models'
prediction_path = 'gpu' + gpu_ind + '/' + parameter_str + '/validation'
# # restore_path = 'gpu001/models/50099_cpkt'

# # optional args
opt_kwargs = {
		'learning_rate': 0.0001#EDIT
}

# # make a trainer for scadec
# # epochs=600
epochs=200
import time
time_start= time.time()
trainer = Trainer_bn(net, batch_size=batch_size, optimizer = "adam", opt_kwargs=opt_kwargs)
path = trainer.train(data_provider, output_path, valid_provider, valid_size, dropout=dropout_rate, training_iters=700, epochs=epochs, display_step=100, save_epoch=20, prediction_path=prediction_path)
time_end = time.time()
print(time_end - time_start)