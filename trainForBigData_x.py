from scadec.unet_bn import Unet_bn
from scadec.train import Trainer_bn

from scadec import image_util

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

para_dict_use = para_dict_21
para_str_use = para_str_21

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

if ('3C' in para_dict_use['Ob']):
    data_channels = 3 
    if 'motion' in para_dict_use['Ob']:
        if ('FULL_SEG' in para_dict_use['Ob']):            
            # data = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
            data1 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
            data2 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
            data3 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_3.mat')
            data  = np.concatenate([data1, data2, data3], axis=0)    
            del(data1, data2, data3)
            vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion.mat')
            

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
        
valid_provider = image_util.SimpleDataProvider(vdata, vtruths)
data_provider = image_util.SimpleDataProvider(data, truths)

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
batch_size = 5 # batch size for training
valid_size = 5  # batch size for validating
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
epochs=200
import time
time_start= time.time()
trainer = Trainer_bn(net, batch_size=batch_size, optimizer = para_dict_use.get('optimizer','adam'), opt_kwargs=opt_kwargs)
path = trainer.train(data_provider, output_path, valid_provider, valid_size, dropout=para_dict_use['Keep'], training_iters=700, epochs=epochs, display_step=100, save_epoch=20, prediction_path=prediction_path)
time_end = time.time()
print(time_end - time_start)
