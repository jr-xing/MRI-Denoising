#%% Import Libraries
from scadec.unet_bn import Unet_bn
from scadec.train import Trainer_bn

from scadec import image_util
from scadec import util

import scipy.io as spio
import numpy as np
import os
import h5py


#%%#################################################
####             PREPARE WORKSPACE               ###
####################################################

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_vis = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_vis; # 0,1,2,3

# here specify the path of the model you want to load

# folder = 'l2_seq_reg_no_drop_0.75'
folder_gpu = '3'
folder = 'l2_3C_motion_reg_no_drop_0.75'
folder_path = 'gpu' + folder_gpu +'/'+folder+'/'
model_path =  'gpu' + folder_gpu + '/' + folder + '/models/final/model.cpkt'


# gpu_ind = '0'
# model_path = 'gpu' + gpu_ind + '/models/60099_cpkt/models/final/model.cpkt'

data_channels = 3
truth_channels = 1

#%%#################################################
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

def gray2rgb(gray_img, mode = 'copy'):
    if mode == 'copy':
        # https://www.zhihu.com/question/266789638
        return np.concatenate((gray_img, gray_img, gray_img), -1)
    elif mode == 'peusodo color':
        # https://stackoverflow.com/questions/10901085/range-values-to-pseudocolor
        pass
    elif mode == 'seq_frame_3':
        # gray_img = vdata        
        imgs = np.zeros(shape = gray_img.shape[:3]+(3,))
        # Previous Slice Channel
        imgs[0,:,:,0] = np.squeeze(gray_img[0,:,:,:])
        imgs[1:,:,:,0] = np.squeeze(gray_img[:-1,:,:,:])
        # Current Slice Channel
        imgs[:,:,:,1] = np.squeeze(gray_img)
        # Next Slice Channel
        imgs[:-1,:,:,2] = np.squeeze(gray_img[1:,:,:,:])
        imgs[-1,:,:,2] = np.squeeze(gray_img[-1,:,:,:])
        return imgs
    elif mode == 'seq_frame_5':
        # gray_img = vdata        
        imgs = np.zeros(shape = gray_img.shape[:3]+(5,))
        # Previous Slices Channel
        # Prev -2 chennel
        imgs[0,:,:,0] = np.squeeze(gray_img[0,:,:,:])
        imgs[1,:,:,0] = np.squeeze(gray_img[0,:,:,:])
        imgs[2:,:,:,0] = np.squeeze(gray_img[:-2:,:,:])
        # Prev -1 chennel
        imgs[1,:,:,1] = np.squeeze(gray_img[1,:,:,:])

        imgs[0:,:,:,0] = np.squeeze(gray_img[0,:,:,:])
        imgs[0:,:,:,1] = np.squeeze(gray_img[0,:,:,:])
        imgs[1:,:,:,0] = np.squeeze(gray_img[1,:,:,:])
        imgs[1:,:,:,1] = np.squeeze(gray_img[:-1,:,:,:])
        # Current Slice Channel
        imgs[:,:,:,2] = np.squeeze(gray_img)
        # Post Slices Channel
        imgs[:-1,:,:,2] = np.squeeze(gray_img[1:,:,:,:])
        imgs[-1,:,:,2] = np.squeeze(gray_img[-1,:,:,:])
        return imgs
    elif mode == 'seq_val_3':
        img_num = int(np.shape(gray_img)[0]/3)
        imgs = np.zeros((img_num,)+np.shape(gray_img)[1:3]+(3,))
        for img_idx in range(img_num):
            sidx = img_idx*3
            imgs[img_idx, :,:,0] = np.squeeze(gray_img[sidx+0,:,:,:])
            imgs[img_idx, :,:,1] = np.squeeze(gray_img[sidx+1,:,:,:])
            imgs[img_idx, :,:,2] = np.squeeze(gray_img[sidx+2,:,:,:])
        return imgs
        


#%%#################################################
####                lOAD MODEL                   ###
####################################################

# set up args for the unet, should be exactly the same as the loading model
kwargs = {
    "layers": 5,
    "conv_times": 2,
    "features_root": 64,
    "filter_size": 3,
    "pool_size": 2,
    "summaries": True
}

net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost="mean_squared_error", **kwargs)


#%%################################################
####                lOAD TRAIN                   ###
####################################################

# #preparing training data
# data_mat = spio.loadmat('train_np/traOb.mat', squeeze_me=True)
# truths_mat = spio.loadmat('train_np/traGt.mat', squeeze_me=True)

# data = data_mat['traOb']
# data = preprocess(data, data_channels) # 4 dimension -> 3 dimension if you do data[:,:,:,1]
# truths = preprocess(truths_mat['traGt'], truth_channels)

# data_provider = image_util.SimpleDataProvider(data, truths)
comp_train_result = True
if comp_train_result:
    data = h5py_mat2npy('train_np/traOb_neigh_motion_mini_25.mat')
    truths = h5py_mat2npy('train_np/traGt_mini_25.mat')
    #data = data[:24,:,:,:]
    #truths = truths[:24,:,:,:]
    # truths_3 = gray2rgb(truths_use, 'seq_frame_3')
    
    data_provider = image_util.SimpleDataProvider(data, truths)


# #%%#################################################
# ####                 lOAD TEST                   ###
# ####################################################

# vdata_mat = spio.loadmat('test_np/testOb.mat'.format(level), squeeze_me=True)
# vtruths_mat = spio.loadmat('test_np/testGt.mat', squeeze_me=True)

# vdata = vdata_mat['testOb']
# vdata = preprocess(vdata, data_channels)
# vtruths = preprocess(vtruths_mat['testGt'], truth_channels)

# valid_provider = image_util.SimpleDataProvider(vdata, vtruths)



#-- Training Data --#
# data_mat = h5py_mat2npy('train_np/traOb.mat')
# truths_mat = h5py_mat2npy('train_np/traGt.mat')

# # data = data_mat['traOb']
# data = preprocess(data_mat, data_channels)    # 4 dimension -> 3 dimension if you do data[:,:,:,1]
# truths = preprocess(truths_mat, truth_channels)

# data_provider = image_util.SimpleDataProvider(data, truths)

#%%#################################################
####                 lOAD TEST                   ###
####################################################
vdata = h5py_mat2npy('valid_np/valOb_neigh_motion.mat')
vtruths = h5py_mat2npy('valid_np/valGt.mat')
# vdata_mat = h5py.File('valid_np/edited/valOb_Crop_160.mat','r')
#vdata = h5py_mat2npy('valid_np/edited/valOb_Crop.mat')
#vtruths = h5py_mat2npy('valid_np/edited/valGt_Crop.mat')
#vdata_3 = gray2rgb(vdata, 'copy')
#vtruths_3 = gray2rgb(vtruths, 'copy')
#vdata_3 = gray2rgb(vdata, 'seq_val_3')
#vtruths_3 = gray2rgb(vtruths, 'seq_val_3')
#vtruths_1 = np.reshape(vtruths_3[:,:,:,1],vtruths_3.shape[:3]+(1,))
#vdata_mat = spio.loadmat('test_np/testOb.mat', squeeze_me=True)
#vtruths_mat = spio.loadmat('test_np/testGt.mat', squeeze_me=True)

#vdata = vdata_mat['testOb']
#vdata = preprocess(vdata, data_channels)
#vtruths = preprocess(vtruths_mat['testGt'], truth_channels)

valid_provider = image_util.SimpleDataProvider(vdata, vtruths)




#%%##################################################
####              	  PREDICT                    ###
####################################################

predicts = []
train_results = []
# num = 5
# num = vdata.shape[0]
num = 25;
# valid_x, valid_y = valid_provider(5)
valid_x, valid_y = valid_provider(num, fix = True)
if comp_train_result:
    data_x, data_y = data_provider(num)




for i in range(num):

    print('')
    # print('')
    print('************* {}/{} *************'.format(i+1, num))
    print('')
    # print('')

    # x_train, y_train = data_provider(5)        
    x_input = valid_x[i:i+1,:,:,:]
    # x_input = np.concatenate((x_input, x_train), axis=0)    
    predict = net.predict(model_path, x_input, 1, False)
    predicts.append(predict[0:1,:,:])
    if comp_train_result:
        x_train = data_x[i:i+1,:,:,:]
        train_result = net.predict(model_path, x_train, 1, False)
        train_results.append(train_result[0:1,:,:])

predicts = np.concatenate(predicts, axis=0)
util.save_mat(predicts, folder_path+'test_{}.mat'.format(folder))
if comp_train_result:
    util.save_mat(train_results, folder_path+'train_result_{}.mat'.format(folder))