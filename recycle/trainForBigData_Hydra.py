from scadec_Hydra.unet_bn_Hydra import Unet_bn
from scadec_Hydra.train_Hydra import Trainer_bn

from scadec_Hydra import image_util

import scipy.io as spio
import numpy as np
import os
import h5py

# Useless Comment

####################################################
####                 FUNCTIONS                   ###
####################################################

# make the data a 4D vector
def preprocess(data, channels):
	nx = data.shape[1]
	ny = data.shape[2]
	return data.reshape((-1, nx, ny, channels))

def h5py_mat2npy(datemat):
    print('Loading '+datemat)
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

def idx_classify(idx_array, n_classes = 8, mode='equally'):
    """ Classify images by slice index(location) """
    class_array = np.ones(np.shape(idx_array), dtype=np.int32)*(-1)
    if mode == 'equally':
        interval = int((idx_array.max() - idx_array.min() + 1)/n_classes)
        for class_idx in range(n_classes):
            idx_start = class_idx * interval + idx_array.min()
            idx_end = (class_idx+1) * interval + idx_array.min()
            class_array[(idx_array>=idx_start)&(idx_array<idx_end)] = class_idx
            # print(idx_end)
        # print((n_classes-1) * interval)        
        return class_array

    elif mode == 'equally_960':
        interval = int((960 - 1 + 1)/n_classes)
        for class_idx in range(n_classes):
            idx_start = class_idx * interval + 1
            idx_end = (class_idx+1) * interval + 1
            class_array[(idx_array>=idx_start)&(idx_array<idx_end)] = class_idx
        return class_array
    elif mode == 'all_zero':
        return class_array*0

    else:
        raise ValueError('Unknown classification method: {}'.format(mode))

def assign_silce_idx(total_count, binSliceStart=1, binSliceEnd=96):
    # if       10:80 in 1:96
    #    idx:  0:71 -> 10:80
    binLength = binSliceEnd-binSliceStart+1
    if total_count % binLength == 0:
        binCount = int(total_count/binLength)
    else:
        raise ValueError("total_count {} and binLength {} Don't match!".format(total_count, binLength))
    return np.array(list(range(binSliceStart, binSliceEnd+1))*binCount)

def add_additional_info_dict_list(ori_dict_list, new_info_list, key_name):
    if ori_dict_list == None:
        ori_dict_list = [{}]*len(new_info_list)

    if len(ori_dict_list) != len(new_info_list):
        raise ValueError("ori dict len {} and info list len {} don't match!".format(len(ori_dict_list), len(new_info_list)))

    for data_idx in range(len(ori_dict_list)):
        ori_dict_list[data_idx][key_name] = new_info_list[data_idx]
    return ori_dict_list   



####################################################
####              HYPER-PARAMETERS               ###
####################################################


# Because we have real & imaginary part of our input, data_channels is set to 2

####################################################
####                DATA LOADING                 ###
####################################################

"""
	here loads all the data we need for training and validating.

"""

#parameter_str = '15-l2_masked-3C_motion-gradient_masked_mid5-reg_no-drop_0.9-FULL_SEG'
para_str_15 = 'Idx_15-Loss_l2_masked_mid5-Loss-gradient_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_15 = {
    'idx':15,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
}

para_str_15_2 = 'Idx_15_2-Loss_l2_masked_mid5-Loss-gradient_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Opti-Clipping'
para_dict_15_2 = {
    'idx':15,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam_clip'
}
# Still got NaN -> use loss cliping (upper_bound for gradient loss) in 

#l2_3C_motion_masked_mid5-Sobel_masked_mid5_w2-reg_no-drop_0.9-FULL_SEG
para_str_16 = 'Idx_16-Loss_l2_masked_mid5-Loss-Sobel_masked_mid5_w2-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_16 = {
    'idx':16,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'Sobel',
        'weight':50,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
}

# l2_3C_motion_masked_mid5-Sobel_masked_mid5-reg_no-drop_0.75-FULL_SEG
para_str_17 = 'Idx_17-Loss_l2_masked_mid5-Loss-Sobel_masked_mid5-Reg_no-Drop_0.75-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_17 = {
    'idx':16,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'Sobel',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.75,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
}


# Re-run 18 and 16 and 17
para_str_19 = 'Idx_19-Loss_l2_masked_mid5-Loss-Sobel_masked_mid5_w2-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_19 = para_dict_16
para_dict_19['GPU_IND'] = '3'

# From 15
para_str_20 = 'Idx_20-Loss_l2_masked_mid5-Loss-gradient_masked_mid5_Clip-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Opti_Clipping'
para_dict_20 = {
    'idx':20,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False,
        'upper_bound': 0.1
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam_clip',
    'GPU_IND':'2'
}

# 	Loss_l2_masked_mid5-Loss-LoG_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG
para_str_21 = 'Idx_21-Loss_l2_masked_mid5-Loss-LoG_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_21 = {
    'idx':21,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'LoG',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False,
        'upper_bound': 0.1
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam_clip',
    'GPU_IND':'3'
}

para_str_22 = 'Idx_22-Loss_l2_masked_mid5-Loss_LoG_masked_mid5_w2-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_22 = {
    'idx':22,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'LoG',
        'weight':50,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'3'
}

para_str_23 = 'Idx_23-Loss_l2_masked_mid5-Loss-gradient_XY_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_23 = {
    'idx':23,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'2'
}

para_str_24 = 'Idx_24-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_24 = {
    'idx':24,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'2'
}

para_str_25 = 'Idx_22-Loss_l2_masked_norm-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_25 = {
    'idx':25,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'norm'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'3'
}

# Redo 24 with lower keep rate
para_str_26 = 'Idx_26-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_26 = {
    'idx':26,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'2'
}

para_str_27 = 'Idx_27-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8-Proc_ero_2'
para_dict_27 = {
    'idx':27,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'proc_dict':{
        'data':{'erosion':{'size':(2,2)}},
        'truth':{}        
    },
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

para_str_28 = 'Idx_28-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8-Proc_ero_3'
para_dict_28 = {
    'idx':28,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'proc_dict':{
        'data':{'erosion':{'size':(3,3)}},
        'truth':{}        
    },
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

para_str_29 = 'Idx_29-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.7-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_29 = {
    'idx':29,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        }],
    'reg':None,
    'Keep':0.7,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

para_str_30 = 'Idx_30-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Proc_ero_4'
para_dict_30 = {
    'idx':30,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'proc_dict':{
        'data':{'erosion':{'size':(4,4)}},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

para_str_31 = 'Idx_31-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_31 = {
    'idx':31,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'n_classes':8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },    
    'epochs':220,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}


# Re-run 29 without gradient in last 20 epochs
# para_str_32 = 'Idx_21-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end_20-Reg_no-Drop_0.7-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_str_32 = 'Idx_32-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end_20-Reg_no-Drop_0.7-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_32 = {
    'idx':32,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_after':200
        }],
    'reg':None,
    'Keep':0.7,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        'structure':'Nagini'# Or Hydra
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,    
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# Re-run 31 with only l2 loss without mask
para_str_33 = 'Idx_33-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_33 = {
    'idx':33,
    'losses':[
        {
        'name':'l2',
        'weight':1}],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'n_classes':8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },    
    'epochs':200,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'2'
}

para_str_34 = 'Idx_34-Loss_l2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra1_8'
para_dict_34 = {
    'idx':34,
    'losses':[
        {
        'name':'l2',
        'weight':1}],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        # 'structure': 'Nagini',
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

para_str_35 = 'Idx_35-Loss_l2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra3_8'
para_dict_35 = {
    'idx':35,
    'losses':[
        {
        'name':'l2',
        'weight':1}],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 3,
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# Re-run 32 with corrected net structure
para_str_36 = 'Idx_36-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra3_8'
para_dict_36 = {
    'idx':36,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':2,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_last':20
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 3,
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# Re-run 34 with corrected net structure(layers = 5) and gradient loss
para_str_37 = 'Idx_37-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_10_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra1_8'
para_dict_37 = {
    'idx':37,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_after':200
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 1,
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# Should be 38!
para_str_38 = 'Idx_37-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_10_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra3_8'
para_dict_38 = {
    'idx':38,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_after':200
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 3,
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

para_str_39 = 'Idx_39-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_10_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_39 = {
    'idx':39,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_after':200
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# re-run 32(i.e. "21") with only 5 additional epochs
para_str_1018 = 'Idx_1018-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_10_invalid_end5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Nagini'
para_dict_1018 = {
    'idx':1018,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        # 'invalid_after':200
        'invalid_last': 5
        }],
    'reg':None,
    'Keep':0.7,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        'structure':'Nagini'# Or Hydra
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':205,    
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# Re-run 39 without gradient mask and last 20
para_str_40 = 'Idx_40-Loss_l2_masked_mid5-Loss-gradient_XY_w_10-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_40 = {
    'idx':40,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# rerun 39 with NMS
para_str_41 = 'Idx_41-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_NMS_w_10_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_41 = {
    'idx':41,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'NMS': True,
        'NMS_window_size': 3,
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# re-run 40 with different gradient weight
para_str_42 = 'Idx_42-Loss_l2_masked_mid5-Loss-gradient_XY_w_30-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_42 = {
    'idx':42,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':30,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# para_dict_1018 = {
#     'idx':1018,
#     'losses':[
#         {
#         'name':'l2',
#         'weight':1,
#         'mask':'mid5'},
#         {
#         'name':'edge',
#         'edge_type':'gradient',
#         'weight':10,
#         'mask':'norm',
#         'mask_before_operate':False,
#         'get_XY':True,
#         'invalid_after':200
#         }],
#     'reg':None,
#     'Keep':0.8,
#     'Ob':'FULL_SEG_3C_motion',
#     'Gt':'FULL_SEG',
#     'kwargs' : {
#         "layers": 4,           # how many resolution levels we want to have
#         "conv_times": 2,       # how many times we want to convolve in each level
#         "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
#         "filter_size": 3,      # filter size used in convolution
#         "pool_size": 2,        # pooling size used in max-pooling
#         "summaries": True,
#         "get_loss_dict": True,
#         "batch_size": 5,
#         "valid_size": 5,
#         'structure': 'Hydra',
#         'neck_len': 2,
#         'n_classes': 16
#     },
#     'proc_dict':{
#         'data':{},
#         'truth':{}        
#     },
#     'epochs':220,
#     'optimizer': 'adam',
#     'server': '2',
#     'GPU_IND':'0'
# }

# para_str_24 = 'Idx_24-Test'
# para_dict_24 = {
#     'idx':24,
#     'losses':[
#         {
#         'name':'l2',
#         'weight':1,
#         'mask':'mid5'},
#         {
#         'name':'edge',
#         'edge_type':'LoG',
#         'weight':1,
#         'mask':'norm'}],
#     'reg':None,
#     'Keep':0.9,
#     'Ob':'FULL_SEG_3C_motion',
#     'Gt':'FULL_SEG',
#     'kwargs' : {
#         "layers": 5,           # how many resolution levels we want to have
#         "conv_times": 2,       # how many times we want to convolve in each level
#         "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
#         "filter_size": 3,      # filter size used in convolution
#         "pool_size": 2,        # pooling size used in max-pooling
#         "summaries": True,
#         "get_loss_dict": True
#     },
#     'optimizer': 'adam',
#     'GPU_IND':'2'
# }

para_dict_use = para_dict_42
para_str_use = para_str_42

import pprint
pprint.pprint('Running '+ para_str_use)
pprint.pprint(para_dict_use)

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = para_dict_use.get('GPU_IND', '3')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind # 0,1,2,3

# -----------------------------------Index------------------------------------------------------ #
# para_idx = parameter_parse[0].split('_')[1]

# -----------------------------------Data------------------------------------------------------- #
# Get data parameters
# ob_para = [para for para in parameter_parse if para.startswith('Ob')][0]
# gt_para = [para for para in parameter_parse if para.startswith('Gt')][0]
# Input



TEST_MODE = False
if TEST_MODE:
    if ('3C' in para_dict_use['Ob']):
        data_channels = 3 
        if 'motion' in para_dict_use['Ob']:
            if ('FULL_SEG' in para_dict_use['Ob']):            
                # data = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')            
                # data = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
                # data_slice_idx_info = add_additional_info_dict_list(None, assign_silce_idx(np.shape(data)[0]), 'slice_idx')
                # data = h5py_mat2npy('../data/valid_np/valOb_neigh_motion.mat')
                data = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_3.mat')
                data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]), mode='equally')
                #data_cls = idx_classify(np.arange(100, 245+1, 5), mode='equally_960')

                
                vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion.mat')
                # vdata_cls = idx_classify(assign_silce_idx(np.shape(vdata)[0]), mode='equally_960')
                vdata_cls = idx_classify(np.arange(100, 245+1, 5), mode='equally_960')

    # Output
    if ('3C' in para_dict_use['Gt']):
        truth_channels = 3
        pass
    else:
        truth_channels = 1
        if ('FULL_SEG' in para_dict_use['Gt']):
            # truths = h5py_mat2npy('train_np/traGt_FULL_SEG_part_1.mat')            
            truths = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_3.mat')
            # truths = h5py_mat2npy('../data/valid_np/valGt.mat')
            vtruths = h5py_mat2npy('../data/valid_np/valGt.mat')
    training_iters = 100
else:
    if ('3C' in para_dict_use['Ob']):
        data_channels = 3 
        if 'motion' in para_dict_use['Ob']:
            if ('FULL_SEG' in para_dict_use['Ob']):            
                # data = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
                data1 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
                data2 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
                data3 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_3.mat')
                data  = np.concatenate([data1, data2, data3], axis=0)    
                #100:5:245
                # data_slice_idx_info = add_additional_info_dict_list(None, assign_silce_idx(np.shape(data)[0]), 'slice_idx')
                # data_slice_cls_info = add_additional_info_dict_list(None, idx_classify(assign_silce_idx(np.shape(data)[0]), mode='equally_960'), 'slice_cls')
                data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]), mode='equally_960')
                del(data1, data2, data3)
                vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion.mat')
                #vdata_slice_idx_info = add_additional_info_dict_list(None, np.arange(100, 245+1, 5), 'slice_idx')
                #vdata_slice_cls_info = add_additional_info_dict_list(None, idx_classify(np.arange(100, 245+1, 5), mode='equally_960'), 'slice_cls')
                vdata_cls = idx_classify(np.arange(100, 245+1, 5), mode='equally_960')

    

    # Output
    if ('3C' in para_dict_use['Gt']):
        truth_channels = 3
        pass
    else:
        truth_channels = 1
        if ('FULL_SEG' in para_dict_use['Gt']):
            # truths = h5py_mat2npy('train_np/traGt_FULL_SEG_part_1.mat')
            truths1 = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_1.mat')
            truths2 = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_2.mat')
            truths3 = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_3.mat')
            truths  = np.concatenate([truths1, truths2, truths3], axis=0)
            del(truths1, truths2, truths3)                
            vtruths = h5py_mat2npy('../data/valid_np/valGt.mat')
    training_iters = 700
        

data_provider = image_util.SimpleDataProvider(data, truths, data_cls = data_cls, data_cls_num=para_dict_use['kwargs'].get('n_classes',1), process_dict = para_dict_use['proc_dict'], onehot_cls=True)
valid_provider = image_util.SimpleDataProvider(vdata, vtruths, data_cls = vdata_cls, data_cls_num=para_dict_use['kwargs'].get('n_classes',1), process_dict = para_dict_use['proc_dict'], onehot_cls=True)


# -----------------------------------Loss------------------------------------------------------- #
losses_dict = para_dict_use['losses']
kwargs = para_dict_use['kwargs']

# if parameter_str == 'l2_3C_motion_masked_masked_Sobel_reg_no_drop_0.9_FULL_SEG':
#     loss = 'mean_squared_error_masked_masked_sobel'
#     cost_kwargs = {
#         "mask_type": "norm",
#         "Sobel_weight": 15
#         }
    
#     dropout_rate = 0.9
    

    



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
net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost_dict_list=losses_dict, **kwargs)


####################################################
####                 TRAINING                    ###
####################################################

# args for training
batch_size = kwargs.get("batch_size", 5) # batch size for training
valid_size = kwargs.get("valid_size", 5)  # batch size for validating
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

# # output paths for results
# output_path = 'gpu' + gpu_ind + '/' + para_str_use + '/models'
# prediction_path = 'gpu' + gpu_ind + '/' + para_str_use + '/validation'
output_path = '../result/gpu' + gpu_ind + '/' + para_str_use + '/models'
prediction_path = '../result/gpu' + gpu_ind + '/' + para_str_use + '/validation'

# # optional args
opt_kwargs = {
		'learning_rate': 0.0001#EDIT
}

# # make a trainer for scadec
# # epochs=600
epochs=para_dict_use.get('epochs', 200)
import time
time_start= time.time()
trainer = Trainer_bn(net, batch_size=batch_size, optimizer = para_dict_use.get('optimizer','adam'), opt_kwargs=opt_kwargs)
path = trainer.train(data_provider, output_path, valid_provider, valid_size, dropout=para_dict_use['Keep'], training_iters=training_iters, epochs=epochs, display_step=100, save_epoch=20, prediction_path=prediction_path)
time_end = time.time()
print(time_end - time_start)
