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

#import cv2
import glob
import numpy as np
from PIL import Image
from scipy import ndimage
import h5py

from scadec_Hydra.util import verbose_print, get_gradients
from scadec_Hydra.crop import random_crop

class BaseDataProvider(object):

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
    
    def __call__(self, n, fix=False):
        if type(n) == int and not fix:
            # X and Y are the images and truths
            train_data, truths, batch_cls, mask = self._next_batch(n)
        elif type(n) == int and fix:
            train_data, truths, batch_cls, mask = self._fix_batch(n)
        elif type(n) == str and n == 'full':
            train_data, truths, batch_cls, mask = self._full_batch() 
        else:
            raise ValueError("Invalid batch_size: "%n)
        
        # print('batch(called)')
        # print(batch_cls)
        return train_data, truths, batch_cls, mask

    def _next_batch(self, n):
        pass

    def _full_batch(self):
        pass


class SimpleDataProvider(BaseDataProvider):
    
    def __init__(self, data, truths, data_cls = None, data_cls_num=None, onehot_cls = False, process_dict = {}, verbose = False, masks=None):
        # additional_info should be list of dicts
        super(SimpleDataProvider, self).__init__()
        # Xing
        self.data = data
        self.truths = truths
        self.size = data.shape[0]
        # self.truth_size = truths.shape[0]
        #self.data = np.float64(data)
        #self.truths = np.float64(truths)
        self.img_channels = self.data[0].shape[2]
        self.truth_channels = self.truths[0].shape[2]
        self.file_count = data.shape[0]
        self.process_dict = process_dict
        # print(data_cls)
        
        # data_cls should be np.array
        self.data_cls = data_cls.astype(np.int32)
        if data_cls_num==None:
            data_cls_num = len(np.unique(self.data_cls))
        else:
            self.data_cls_num = data_cls_num
        
        self.onehot_cls = onehot_cls
        self.verbose = verbose
        self.masks = masks


    def _next_batch(self, n):
        idx = np.random.choice(self.file_count, n, replace=False)
        img = self.data[idx[0]]
        nx = img.shape[0]
        ny = img.shape[1]
        X = np.zeros((n, nx, ny, self.img_channels))
        Y = np.zeros((n, nx, ny, self.truth_channels))

        # Get data        
        for i in range(n):
            X[i] = self._process_data(self.data[idx[i]])
            Y[i] = self._process_truths(self.truths[idx[i]])
                
        # Get labels
        if self.onehot_cls:
            batch_cls = np.zeros([n, self.data_cls_num])
        else:
            batch_cls = np.zeros(n)
        for i in range(n):
            if self.onehot_cls:
                batch_cls[i, self.data_cls[idx[i]]] = 1
            else:    
                batch_cls[i] = self.data_cls[idx[i]]

        # Get masks
        masks = np.ones((n, nx, ny, self.truth_channels))
        if self.masks is not None:
            for i in range(n):                
                # masks[i] = self.masks[idx[i]]
                # Normalize masks
                mask_min = np.min(self.masks[idx[i]])
                mask_max = np.max(self.masks[idx[i]] - mask_min)
                if mask_max != 0:
                    masks[i] = (self.masks[idx[i]]-mask_min)/mask_max
                if np.sum(np.isnan(masks[i]))!=0:
                    print('NAN!')
                    print('MASK_MIN: ' + str(mask_min))
                    print('MASK_MAX: ' + str(mask_max))

            # mask = predict_mask()

        # print('(In data Provider)')
        # print('onehot_cls:')
        # print(self.onehot_cls)
        # print('np.shape(batch_cls)')
        # print(np.shape(batch_cls))
        # verbose_print('feed batch_cls in dataProvider: ',self.verbose)
        # verbose_print(batch_cls,self.verbose)
        return X, Y, batch_cls, masks

    def _fix_batch(self, n):
        # first n data
        img = self.data[0]
        nx = img.shape[0]
        ny = img.shape[1]
        X = np.zeros((n, nx, ny, self.img_channels))
        Y = np.zeros((n, nx, ny, self.truth_channels))            

        # Get images
        for i in range(n):
            X[i] = self._process_data(self.data[i])
            Y[i] = self._process_truths(self.truths[i])
        
        # Get labels
        if self.onehot_cls:
            batch_cls = np.zeros([n, self.data_cls_num])
        else:
            batch_cls = np.zeros(n)
        if self.data_cls_num != 1:
            for i in range(n):
                if self.onehot_cls:
                    batch_cls[i, self.data_cls[i]] = 1
                else:    
                    batch_cls[i] = self.data_cls[i]
        
        # Get masks
        masks = np.ones((n, nx, ny, self.truth_channels))
        if self.masks is not None:
            for i in range(n):
                masks[i] = self.masks[i]
                # Normalize masks
                mask_min = np.min(self.masks[i])
                mask_max = np.max(self.masks[i] - mask_min)
                if mask_max != 0:
                    masks[i] = (self.masks[i]-mask_min)/mask_max
                if np.sum(np.isnan(masks[i]))!=0:
                    print('NAN!')
                    print('MASK_MIN: ' + str(mask_min))
                    print('MASK_MAX: ' + str(mask_max))
                # print(np.isnan(masks[i]))

        return X, Y, batch_cls, masks

    def _full_batch(self):
        return self._fix_batch(self.file_count)
        # return self.data, self.truths

    def _process_truths(self, truth):
        # normalization by channels
        truth = np.clip(np.fabs(truth), self.a_min, self.a_max)        
        for channel in range(self.truth_channels):
            truth[:,:,channel] = self._process_single_img_single_chan(truth[:,:,channel], 'truth')
        return truth

    def _process_data(self, data):
        # normalization by channels
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        for channel in range(self.img_channels):
            data[:,:,channel] = self._process_single_img_single_chan(data[:,:,channel], 'data')
        return data
    
    def _process_single_img_single_chan(self, gray_img, data_or_truth = 'data', normalize=True):
        for proc_type, proc_para in self.process_dict[data_or_truth].items():
            if proc_type == 'erosion':
                gray_img = ndimage.grey_erosion(gray_img, size=proc_para['size']).astype(gray_img.dtype)
            else:
                raise ValueError('Unknown processing operation: {}'.format(proc_type))

        if normalize:
            gray_img -= np.amin(gray_img[:,:])
            gray_img /= np.amax(gray_img[:,:])
        return gray_img

# import matplotlib.pyplot as plt

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img


def concat_n_images(image_mat):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i in range(image_mat.shape[0]):
        # img = plt.imread(img_path)[:,:,:3]
        img = image_mat[i,...]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output

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
    """ Classify images by slice index(location), idx from 1 """
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
    elif mode == 'equally_96':
        interval = int((96 - 1 + 1)/n_classes)
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

# def add_additional_info_dict_list(ori_dict_list, new_info_list, key_name):
#     if ori_dict_list == None:
#         ori_dict_list = [{}]*len(new_info_list)

#     if len(ori_dict_list) != len(new_info_list):
#         raise ValueError("ori dict len {} and info list len {} don't match!".format(len(ori_dict_list), len(new_info_list)))

#     for data_idx in range(len(ori_dict_list)):
#         ori_dict_list[data_idx][key_name] = new_info_list[data_idx]
#     return ori_dict_list 

def get_data_provider(para_dict_use, mode = 'train', DEBUG_MODE = False):
    # data_cls_num = para_dict_use['kwargs'].get('n_classes',1)
    print(para_dict_use['kwargs'])
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
                        data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]),n_classes = data_cls_num, mode='equally')
                        #data_cls = idx_classify(np.arange(100, 245+1, 5), mode='equally_960')
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
                        data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]), n_classes = data_cls_num, mode='equally')
                        # print('Observation_size')
                        # print(np.shape(data)[0])
                        # print('data_cls (at image_utils.py): ')
                        # print(data_cls)
                        del(data1, data2, data3)
                        vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion.mat')
                        vdata_idx = np.arange(100, 245+1, 5)%96
                        vdata_idx[vdata_idx==0] = 96
                        # vdata_cls = idx_classify(np.arange(100, 245+1, 5),n_classes = data_cls_num, mode='equally_96')    
                        vdata_cls = idx_classify(vdata_idx,n_classes = data_cls_num, mode='equally_96')    
                    elif mode == 'test' or mode == 'valid':
                        data = None
                        vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion_FULL.mat')
                        vdata_cls = idx_classify(assign_silce_idx(np.shape(vdata)[0]),n_classes = data_cls_num, mode = 'equally')
                        # vdata_cls = idx_classify(np.arange(100, 245+1, 5), mode='equally_960')
                    elif mode == 'test_on_train':
                        data=None
                        vdata = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')[np.arange(99,245,5),:,:,:]
                        # idx = np.arange(99,245,5)                        
                        vdata_idx = np.arange(100, 245+1, 5)%96
                        vdata_idx[vdata_idx==0] = 96
                        # vdata_cls = idx_classify(np.arange(100, 245+1, 5),n_classes = data_cls_num, mode='equally_96')    
                        vdata_cls = idx_classify(vdata_idx,n_classes = data_cls_num, mode='equally_96')    
                        # vdata_cls = idx_classify(assign_silce_idx(np.shape(vdata)[0]),n_classes = data_cls_num, mode = 'equally')

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
                    # del(truths1, truths2, truths3)                
                    vtruths = h5py_mat2npy('../data/valid_np/valGt.mat')
                elif mode == 'test' or mode == 'valid':
                    truths = None
                    vtruths = h5py_mat2npy('../data/valid_np/valGt_FULL.mat')
                elif mode == 'test_on_train':
                    vtruths = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_1.mat')[np.arange(99,245,5),:,:,:]                    

        training_iters = 700
        # training_iters = 50
    ignore_classes = para_dict_use.get('ignore_classes', [])
    # def find_min_empty_cls(cls_array, startCls = 0):
    #     currentCls = startCls
    #     while np.sum(cls_array[cls_array==currentCls]) != 0:
    #         currentCls += 1
    #     return currentCls

    # print(vdata_cls)
    if para_dict_use.get('preMask', False):
        print('Loading vmasks...')
        vdata_masks = np.load('../data/masks/vdata_FULL_SEG_28_28_masks.npy')
        # vdata_masks = predict_masks(vtruths, savePatches=False, saveYPre=True, saveMasks=True, saveName='vdata_'+ para_dict_use['Gt'])
        # vdata_masks = predict_masks(vtruths, savePatches=False, loadYPre=True, saveMasks=True, saveName='vdata_'+ para_dict_use['Gt'])

        if data is not None:
            # data_masks_1 = predict_masks(truths1, savePatches=False, saveYPre=True, saveMasks=True, saveName='data_'+ para_dict_use['Gt'] + '_1')
            # data_masks_2 = predict_masks(truths2, savePatches=False, saveYPre=True, saveMasks=True, saveName='data_'+ para_dict_use['Gt'] + '_2')
            # data_masks_3 = predict_masks(truths3, savePatches=False, saveYPre=True, saveMasks=True, saveName='data_'+ para_dict_use['Gt'] + '_3')
            

            # data_masks = predict_masks(truths, savePatches=False, saveYPre=True, saveMasks=True, saveName='data_'+ para_dict_use['Gt'])
            print('Loading masks...')
            if DEBUG_MODE:
                data_masks = np.load('../data/masks_debug/data_FULL_SEG_28_28_masks.npy')
            else:
                data_masks_1 = np.load('../data/masks/data_FULL_SEG_1_28_28_masks.npy')
                data_masks_2 = np.load('../data/masks/data_FULL_SEG_2_28_28_masks.npy')
                data_masks_3 = np.load('../data/masks/data_FULL_SEG_3_28_28_masks.npy')

                data_masks  = np.concatenate([data_masks_1, data_masks_2, data_masks_3], axis=0)    

            # data_masks = predict_masks(truths, savePatches=False, loadYPre=True, saveMasks=True, saveName='data_'+ para_dict_use['Gt'])
        else:
            data_masks = None    

    for idx, ignore_class in enumerate(ignore_classes):
        ignore_class -= idx
        data = data[data_cls != ignore_class,:,:,:]
        vdata = vdata[vdata_cls != ignore_class,:,:,:]        
        truths = truths[data_cls != ignore_class,:,:,:]
        vtruths = vtruths[vdata_cls != ignore_class,:,:,:]
        data_masks = data_masks[data_cls != ignore_class, :, :, :]
        vdata_masks = vdata_masks[vdata_cls != ignore_class, :, :, :]
        data_cls = data_cls[data_cls != ignore_class]
        vdata_cls = vdata_cls[vdata_cls != ignore_class]

        for reAssignIdx in range(ignore_class+1, data_cls_num):
            data_cls[data_cls == reAssignIdx] -= 1
            vdata_cls[vdata_cls == reAssignIdx] -= 1
        data_cls_num -= 1
        # for cls in
        # if ignore_class != data_cls_num:
        #     data_cls[data_cls == 0]
    # print(vdata_cls)

    
    if mode == 'train':
        data_provider = SimpleDataProvider(data, truths, data_cls = data_cls, data_cls_num=data_cls_num, process_dict = para_dict_use['proc_dict'], onehot_cls=True, verbose=False, masks = data_masks)        
    elif mode == 'test' or mode == 'valid' or mode == 'test_on_train':
        data_provider = None
    else:
        raise ValueError('Unknow mode' + mode)
    
    
    valid_provider = SimpleDataProvider(vdata, vtruths, data_cls = vdata_cls, data_cls_num=data_cls_num, process_dict = para_dict_use['proc_dict'], onehot_cls=True, verbose=False, masks = vdata_masks)
    return data_provider, valid_provider, data_channels, truth_channels, training_iters


import tensorflow as tf
import time
from tqdm import tqdm
import pickle
# import numpy as np
# def predict_masks_old3(test_imgs, crop_num = 5000, model_path = './scadec_Hydra/maskCNN/CNN_mask-2000.meta',
#                     saveName = 'patches', loadName = 'patches',
#                     savePatches = False, loadPatches = False,
#                     saveYPre = False, loadYPre = False,
#                     saveMasks = False, loadMaskes = False):#, savePath = None):
#     print('Computing masks...')
#     startTime = time.time()

#     # 1. Get croped patches    
#     img_n, img_h, img_w, img_c = np.shape(test_imgs)
#     masks = np.zeros(np.shape(test_imgs))
    
#     cropSize = (28,28)    
    

#     with tf.Session() as sess:
#         # Get feed data(img+gradient)
#         new_saver = tf.train.import_meta_graph(model_path)
#         new_saver.restore(sess, '.'.join(model_path.split('.')[:-1]))
        
#         graph = tf.get_default_graph()
#         xs = graph.get_operation_by_name('xs').outputs[0]
#         pred = graph.get_operation_by_name('pred').outputs[0]

#         savefile_img_num = 2000
#         for imgIdxInAll in tqdm(range(img_n),ncols=75):
#             # Get patches of {savefile_img_num} imgs
#             imgIdx = imgIdxInAll % savefile_img_num
#             # img_n_file = 
#             if load and imgIdx == 0:
#                 patches, boxeses = np.load('./scadec_Hydra/maskCNN/pre_computed/{}_{}_{}_patches_{}.pkl'.format(loadName, cropSize[0], cropSize[1], int((imgIdx+1)/savefile_img_num)))
#             else:
#                 patches = np.zeros([savefile_img_num*crop_num, cropSize[0], cropSize[1], 2])
#                 boxeses = []
#                 for imgIdx in range(img_n):
#                     img = np.squeeze(test_imgs[imgIdx])
#                     img_patches, boxes = random_crop(img, crop_size = cropSize, crop_num = crop_num,
#                                 save = False, saveMode='npy')        
#                     img_patches = np.reshape(img_patches,list(np.shape(img_patches))+[1])
#                     img_patches_grad = np.reshape(get_gradients(img_patches[:,:,:,0]),(len(boxes), 28, 28, 1))
                    
#                     datum = np.zeros((len(boxes), 28,28,2))
#                     for idx in range(len(boxes)):
#                         datum[idx, :, :, :] = np.concatenate([img_patches[idx,:,:], img_patches_grad[idx,:,:]], axis = 2)                
                    
#                     patches[imgIdx*crop_num:(imgIdx+1)*crop_num, :, :, :] = datum
#                     boxeses.append(boxes)
            
#             if save and imgIdx == 0:
#                 # print('Saving...')
#                 np.save('./scadec_Hydra/maskCNN/pre_computed/{}_{}_{}_patches_{}.pkl'.format(saveName, cropSize[0], cropSize[1], int((imgIdx+1)/2000)), [patches, boxeses])

#             # Get Predictions
#             if loadYPre:
#                 y_pre_raw = np.load('./scadec_Hydra/maskCNN/pre_computed/{}_{}_{}_ypre_boxes_{}.pkl'.format(saveName, cropSize[0], cropSize[1], int((imgIdx+1)/2000)))
#             else:
#                 # y_pre_raw = np.zeros()
#                 y_pre_raws = []
#                 for imgIdx in tqdm(range(img_n),ncols=75):
#                     y_pre_raws.append(sess.run(pred, feed_dict={xs:patches[imgIdx*crop_num:(imgIdx+1)*crop_num]}))
#             if saveYPre:
#                 np.save('./scadec_Hydra/maskCNN/pre_computed/{}_{}_{}_ypre_boxes_{}.pkl'.format(saveName, cropSize[0], cropSize[1], int((imgIdx+1)/2000), y_pre_raws))
            
#             # Get masks            
#             for imgIdx in range(img_n):
#                 y_pre = y_pre_raws[imgIdx][:,0]>0.9
#                 boxes = boxeses[imgIdx]
#                 mask = np.zeros(np.shape(img))
#                 for idx, box in enumerate(boxes):
#                     if y_pre[idx] == True:
#                         mask[box[0]:box[1],box[2]:box[3]] += 0.01
#                 masks[imgIdx] = np.reshape(mask, [img_h, img_w, img_c])
#             endTime = time.time()
#         print('Finished computing masks for {} within {} mins for {} images!'.format(saveName, (endTime - startTime)/60, img_n))
#     return masks

def predict_masks(test_imgs, crop_num = 5000, model_path = './scadec_Hydra/maskCNN/CNN_mask-2000.meta',
                    save_path = '..data/masks/',
                    saveName = 'patches', loadName = 'patches',
                    savePatches = False, loadPatches = False,
                    saveYPre = False, loadYPre = False,
                    saveMasks = False, loadMaskes = False):#, savePath = None):
    # import os
    # print(os.getcwd())
    print('Computing masks...')
    startTime = time.time()

    # 1. Get croped patches    
    img_n, img_h, img_w, img_c = np.shape(test_imgs)
    masks = np.zeros(np.shape(test_imgs))
    print(np.shape(masks))
    
    cropSize = (28,28)    
    savefile_img_num = 2000

    with tf.Session() as sess:
        # Get feed data(img+gradient)
        new_saver = tf.train.import_meta_graph(model_path)
        new_saver.restore(sess, '.'.join(model_path.split('.')[:-1]))
        
        graph = tf.get_default_graph()
        xs = graph.get_operation_by_name('xs').outputs[0]
        pred = graph.get_operation_by_name('pred').outputs[0]

        # Get patches
        print('Getting patches...')
        if loadPatches:
            patches, boxeses = np.load(save_path + '{}_{}_{}_patches_boxeses.npy'.format(loadName, cropSize[0], cropSize[1]))
            # with open('./scadec_Hydra/maskCNN/pre_computed/{}.pkl'.format(loadName), 'rb') as f:  # Python 3: open(..., 'wb')
            #     data = pickle.load(f)
        elif (not loadYPre) and (not loadMaskes):
            patches = np.zeros([img_n*crop_num, cropSize[0], cropSize[1], 2])
            boxeses = []
            # for preTraIdx in tqdm(range(500), ncols=75):
            for imgIdx in tqdm(range(img_n),ncols=75):
                img = np.squeeze(test_imgs[imgIdx])
                img_patches, boxes = random_crop(img, crop_size = cropSize, crop_num = crop_num,
                            save = False, saveMode='npy')        
                img_patches = np.reshape(img_patches,list(np.shape(img_patches))+[1])
                img_patches_grad = np.reshape(get_gradients(img_patches[:,:,:,0]),(len(boxes), 28, 28, 1))
                
                datum = np.zeros((len(boxes), 28,28,2))
                for idx in range(len(boxes)):
                    datum[idx, :, :, :] = np.concatenate([img_patches[idx,:,:], img_patches_grad[idx,:,:]], axis = 2)                
                # data.append(datum)
                patches[imgIdx*crop_num:(imgIdx+1)*crop_num, :, :, :] = datum
                boxeses.append(boxes)
        
        if savePatches:
            print('Saving...')
            # np.save('./scadec_Hydra/maskCNN/pre_computed/{}_{}_{}_patches_boxeses'.format(saveName, cropSize[0], cropSize[1]), [patches, boxeses])
            np.save(save_path + '{}_{}_{}_patches'.format(saveName, cropSize[0], cropSize[1]), patches)

        # Get Predictions
        print('Predicting...')
        if loadYPre:
            y_pre_raws = np.load(save_path + '{}_{}_{}_ypre.npy'.format(saveName, cropSize[0], cropSize[1]))
            boxeses = np.load(save_path + '{}_{}_{}_boxeses.npy'.format(saveName, cropSize[0], cropSize[1]))
        elif not loadMaskes:
            y_pre_raws = []
            for imgIdx in tqdm(range(img_n),ncols=75):
                y_pre_raws.append(sess.run(pred, feed_dict={xs:patches[imgIdx*crop_num:(imgIdx+1)*crop_num]}))
        
            # a = 1
        if saveYPre:
            np.save(save_path + '{}_{}_{}_ypre.npy'.format(saveName, cropSize[0], cropSize[1]), y_pre_raws)
            np.save(save_path + '{}_{}_{}_boxeses.npy'.format(saveName, cropSize[0], cropSize[1]), boxeses)
        
        # Get masks
        # y_pres = []
        print('Computing masks...')
        for imgIdx in tqdm(range(img_n),ncols=75):
            y_pre = y_pre_raws[imgIdx][:,0]>0.9
            boxes = boxeses[imgIdx]
            mask = np.zeros(np.shape(img))
            for idx, box in enumerate(boxes):
                if y_pre[idx] == True:
                    mask[box[0]:box[1],box[2]:box[3]] += 0.01
            masks[imgIdx] = np.reshape(mask, [img_h, img_w, img_c])
        if saveMasks:
            np.save(save_path + '{}_{}_{}_masks'.format(saveName, cropSize[0], cropSize[1]), masks)
        endTime = time.time()
        print('Finished computing masks for {} within {} mins for {} images!'.format(saveName, (endTime - startTime)/60, img_n))
    return masks

def predict_masks_old2(test_imgs, crop_num = 5000, model_path = './scadec_Hydra/maskCNN/CNN_mask-2000.meta'):
    print('Computing masks...')
    startTime = time.time()

    # 1. Get croped patches    
    img_n, img_h, img_w, img_c = np.shape(test_imgs)
    masks = np.zeros(np.shape(test_imgs))
    
    cropSize = (28,28)    
    

    with tf.Session() as sess:
        # Get feed data(img+gradient)
        new_saver = tf.train.import_meta_graph(model_path)
        new_saver.restore(sess, '.'.join(model_path.split('.')[:-1]))
        
        graph = tf.get_default_graph()
        xs = graph.get_operation_by_name('xs').outputs[0]
        pred = graph.get_operation_by_name('pred').outputs[0]

        # Get patches
        print('Getting patches...')
        data = []
        for imgIdx in range(img_n):
            img = np.squeeze(test_imgs[imgIdx])
            img_patches, boxes = random_crop(img, crop_size = cropSize, crop_num = crop_num,
                        save = False, saveMode='npy')        
            img_patches = np.reshape(img_patches,list(np.shape(img_patches))+[1])
            img_patches_grad = np.reshape(get_gradients(img_patches[:,:,:,0]),(len(boxes), 28, 28, 1))
            
            datum = np.zeros((len(boxes), 28,28,2))
            for idx in range(len(boxes)):
                datum[idx, :, :, :] = np.concatenate([img_patches[idx,:,:], img_patches_grad[idx,:,:]], axis = 2)                
            data.append(datum)
        
        # Get Predictions
        print('Predicting...')
        y_pre_raws = []
        for imgIdx in range(img_n):
            y_pre_raws.append(sess.run(pred, feed_dict={xs:data[imgIdx]}))
        
        # Get masks
        # y_pres = []
        print('Computing masks...')
        for imgIdx in range(img_n):
            y_pre = y_pre_raws[imgIdx][:,0]>0.9
            mask = np.zeros(np.shape(img))
            for idx, box in enumerate(boxes):
                if y_pre[idx] == True:
                    mask[box[0]:box[1],box[2]:box[3]] += 0.01
            masks[imgIdx] = np.reshape(mask, [img_h, img_w, img_c])
        endTime = time.time()
        print('Finished computing masks within {%.2f} mins for {} images!'.format((endTime - startTime)/60, img_n))
    return masks

        # masks[imgIdx] = np.reshape(predict_mask(np.squeeze(test_imgs[imgIdx]), crop_num, model_path), (img_h, img_w, img_c))
    # test_img_patches, boxes = random_crop(test_img, crop_size = cropSize, crop_num = crop_num,
    #                   save = False, saveMode='npy')        
    # test_img_patches = np.reshape(test_img_patches,list(np.shape(test_img_patches))+[1])
    # test_img_patches_grad = np.reshape(get_gradients(test_img_patches[:,:,:,0]),(len(boxes), 28, 28, 1))
    # test_data = np.zeros((len(boxes), 28,28,2))



    
def predict_masks_old(test_imgs, crop_num = 5000, model_path = './scadec_Hydra/maskCNN/CNN_mask-2000.meta'):
    print('computing masks...')
    img_n, img_h, img_w, img_c = np.shape(test_imgs)
    masks = np.zeros(np.shape(test_imgs))
    for imgIdx in range(img_n):
        masks[imgIdx] = np.reshape(predict_mask(np.squeeze(test_imgs[imgIdx]), crop_num, model_path), (img_h, img_w, img_c))
    return np.reshape(masks, [img_n, img_h, img_w, 1])

def predict_mask(test_img, crop_num = 5000, model_path = './scadec_Hydra/maskCNN/CNN_mask-2000.meta'):
    cropSize = (28,28)
    if len(np.shape(test_img)) == 3:
        test_img = test_img[:,:,0]
    
    test_img_patches, boxes = random_crop(test_img, crop_size = cropSize, crop_num = crop_num,
                      save = False, saveMode='npy')        
    test_img_patches = np.reshape(test_img_patches,list(np.shape(test_img_patches))+[1])
    test_img_patches_grad = np.reshape(get_gradients(test_img_patches[:,:,:,0]),(len(boxes), 28, 28, 1))
    test_data = np.zeros((len(boxes), 28,28,2))
            
    for idx in range(len(boxes)):
        test_data[idx, :, :, :] = np.concatenate([test_img_patches[idx,:,:], test_img_patches_grad[idx,:,:]], axis = 2)
    
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_path)
        new_saver.restore(sess, '.'.join(model_path.split('.')[:-1]))
        
        graph = tf.get_default_graph()
        xs = graph.get_operation_by_name('xs').outputs[0]
        pred = graph.get_operation_by_name('pred').outputs[0]            
        y_pre_raw = sess.run(pred, feed_dict={xs:test_data})
    
    # print(y_pre_raw[:50,0])
    y_pre = y_pre_raw[:,0]>0.9
    mask = np.zeros(np.shape(test_img))
    for idx, box in enumerate(boxes):
        if y_pre[idx] == True:
            mask[box[0]:box[1],box[2]:box[3]] += 0.01
    
    return mask