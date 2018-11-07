from scadec_Hydra.util import verbose_print, get_gradients
from scadec_Hydra.crop import random_crop
from configs import para_dict_use_train, para_str_use_train

import glob
import numpy as np
from PIL import Image
from scipy import ndimage
import h5py
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
def get_data(para_dict_use, mode = 'train', DEBUG_MODE = False):
    # data_cls_num = para_dict_use['kwargs'].get('n_classes',1)
    if type(para_dict_use['kwargs']['structure']) == str:
        data_cls_num = para_dict_use['kwargs'].get('n_classes',1)
    else:
        data_cls_num = para_dict_use['kwargs']['structure'].get('n_classes', 1)
    # print(data_cls_num)
    if DEBUG_MODE:
        # Observation
        if ('3C' in para_dict_use['Ob']):
            data_channels = 3 
            if 'motion' in para_dict_use['Ob']:
                if ('FULL_SEG' in para_dict_use['Ob']):            
                    if mode == 'train':
                        data = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
                    elif mode == 'test' or mode == 'valid':
                        data = None
                    
                    vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion.mat')
                    # vdata_cls = idx_classify(assign_silce_idx(np.shape(vdata)[0]), mode='equally_960')
                    vdata_idx = np.arange(100, 245+1, 5)%96
                    vdata_idx[vdata_idx==0] = 96
                    #vdata_cls = idx_classify(np.arange(100, 245+1, 5)%96,n_classes = data_cls_num, mode='equally_96')
                    vdata_cls = idx_classify(vdata_idx,n_classes = data_cls_num, mode='equally_96')
                        

        # Truth (Target)
        if ('3C' in para_dict_use['Gt']):
            truth_channels = 3
            pass
        else:
            truth_channels = 1
            if ('FULL_SEG' in para_dict_use['Gt']):
                if mode == 'train':       
                    # truths = h5py_mat2npy('train_np/traGt_FULL_SEG_part_1.mat')            
                    truths = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_2.mat')
                    # truths = h5py_mat2npy('../data/valid_np/valGt.mat')
                    vtruths = h5py_mat2npy('../data/valid_np/valGt.mat')
                elif mode == 'test' or mode == 'valid':
                    truths = None
                    vtruths = h5py_mat2npy('../data/valid_np/valGt.mat')


        training_iters = 50
    else:
        # Obeservation / input noisy data
        if ('3C' in para_dict_use['Ob']):
            data_channels = 3 
            if 'motion' in para_dict_use['Ob']:
                if ('FULL_SEG' in para_dict_use['Ob']):   
                    if mode == 'train':         
                        # data = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
                        data1 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
                        data2 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
                        data3 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_3.mat')
                        data  = np.concatenate([data1, data2, data3], axis=0)    
                        del(data1, data2, data3)
                        vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion.mat')
                        vdata_idx = np.arange(100, 245+1, 5)%96
                        vdata_idx[vdata_idx==0] = 96
                    elif mode == 'test' or mode == 'valid':
                        data = None
                        vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion_FULL.mat')
                    elif mode == 'test_on_train':
                        data=None
                        vdata = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')[np.arange(99,245,5),:,:,:]                       

        # Ground truth / Target clean data
        if ('3C' in para_dict_use['Gt']):
            truth_channels = 3
            pass
        else:
            truth_channels = 1
            if ('FULL_SEG' in para_dict_use['Gt']):
                if mode == 'train':
                    # truths = h5py_mat2npy('train_np/traGt_FULL_SEG_part_1.mat')
                    truths1 = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_1.mat')
                    truths2 = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_2.mat')
                    truths3 = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_3.mat')
                    truths  = np.concatenate([truths1, truths2, truths3], axis=0)
                    del(truths1, truths2, truths3)                
                    vtruths = h5py_mat2npy('../data/valid_np/valGt.mat')
                elif mode == 'test' or mode == 'valid':
                    truths = None
                    vtruths = h5py_mat2npy('../data/valid_np/valGt_FULL.mat')
                elif mode == 'test_on_train':
                    vtruths = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_1.mat')[np.arange(99,245,5),:,:,:]                    


    
    return data, vdata, truths, vtruths


#%% 1. Getting patches
crop_num = 5000
cropSize = (28,28)

imgs = get_data(para_dict_use_train, DEBUG_MODE=False)

img_n, img_h, img_w, img_c = np.shape(imgs)
data = np.zeros([img_n*crop_num, img_h, img_w, img_c])
for imgIdx in range(img_n):
    img = np.squeeze(imgs[imgIdx])
    img_patches, boxes = random_crop(img, crop_size = cropSize, crop_num = crop_num,
                save = False, saveMode='npy')        
    img_patches = np.reshape(img_patches,list(np.shape(img_patches))+[1])
    img_patches_grad = np.reshape(get_gradients(img_patches[:,:,:,0]),(len(boxes), 28, 28, 1))
    
    datum = np.zeros((len(boxes), 28,28,2))
    for idx in range(len(boxes)):
        datum[idx, :, :, :] = np.concatenate([img_patches[idx,:,:], img_patches_grad[idx,:,:]], axis = 2)                
    data[img_n]


model_path = './scadec_Hydra/maskCNN/CNN_mask-2000.meta'