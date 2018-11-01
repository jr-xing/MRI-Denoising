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

from scadec_Hydra.util import verbose_print

class BaseDataProvider(object):

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
    
    def __call__(self, n, fix=False):
        if type(n) == int and not fix:
            # X and Y are the images and truths
            train_data, truths, batch_cls = self._next_batch(n)
        elif type(n) == int and fix:
            train_data, truths, batch_cls = self._fix_batch(n)
        elif type(n) == str and n == 'full':
            train_data, truths, batch_cls = self._full_batch() 
        else:
            raise ValueError("Invalid batch_size: "%n)
        
        # print('batch(called)')
        # print(batch_cls)
        return train_data, truths, batch_cls

    def _next_batch(self, n):
        pass

    def _full_batch(self):
        pass


class SimpleDataProvider(BaseDataProvider):
    
    def __init__(self, data, truths, data_cls = None, data_cls_num=None, onehot_cls = False, process_dict = {}, verbose = False):
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
        print(data_cls)
        
        # data_cls should be np.array
        self.data_cls = data_cls.astype(np.int32)
        if data_cls_num==None:
            data_cls_num = len(np.unique(self.data_cls))
        else:
            self.data_cls_num = data_cls_num
        
        self.onehot_cls = onehot_cls
        self.verbose = verbose


    def _next_batch(self, n):
        idx = np.random.choice(self.file_count, n, replace=False)
        img = self.data[idx[0]]
        nx = img.shape[0]
        ny = img.shape[1]
        X = np.zeros((n, nx, ny, self.img_channels))
        Y = np.zeros((n, nx, ny, self.truth_channels))

        if self.onehot_cls:
            batch_cls = np.zeros([n, self.data_cls_num])
        else:
            batch_cls = np.zeros(n)

        if self.data_cls_num == 1:
            for i in range(n):
                X[i] = self._process_data(self.data[idx[i]])
                Y[i] = self._process_truths(self.truths[idx[i]])
        else:            
            # verbose_print('next batch indices:'， self.verbose)
            verbose_print('next batch indices:',self.verbose)
            verbose_print(idx, self.verbose)
            for i in range(n):
                X[i] = self._process_data(self.data[idx[i]])
                Y[i] = self._process_truths(self.truths[idx[i]])
                if self.onehot_cls:
                    batch_cls[i, self.data_cls[idx[i]]] = 1
                else:    
                    batch_cls[i] = self.data_cls[idx[i]]
        
        # print('(In data Provider)')
        # print('onehot_cls:')
        # print(self.onehot_cls)
        # print('np.shape(batch_cls)')
        # print(np.shape(batch_cls))
        verbose_print('feed batch_cls in dataProvider: ',self.verbose)
        verbose_print(batch_cls,self.verbose)
        return X, Y, batch_cls

    def _fix_batch(self, n):
        # first n data
        img = self.data[0]
        nx = img.shape[0]
        ny = img.shape[1]
        X = np.zeros((n, nx, ny, self.img_channels))
        Y = np.zeros((n, nx, ny, self.truth_channels))
        
        if self.onehot_cls:
            batch_cls = np.zeros([n, self.data_cls_num])
        else:
            batch_cls = np.zeros(n)

        if self.data_cls_num == 1:
            for i in range(n):
                X[i] = self._process_data(self.data[i])
                Y[i] = self._process_truths(self.truths[i])
        else:            
            for i in range(n):
                X[i] = self._process_data(self.data[i])
                Y[i] = self._process_truths(self.truths[i])
                # print('self.data_cls[i]')
                # print(type(self.data_cls[i]))
                # print(self.data_cls[i])
                if self.onehot_cls:
                    batch_cls[i, self.data_cls[i]] = 1
                else:    
                    batch_cls[i] = self.data_cls[i]
        
        return X, Y, batch_cls

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
                    del(truths1, truths2, truths3)                
                    vtruths = h5py_mat2npy('../data/valid_np/valGt.mat')
                elif mode == 'test' or mode == 'valid':
                    truths = None
                    vtruths = h5py_mat2npy('../data/valid_np/valGt_FULL.mat')
                elif mode == 'test_on_train':
                    vtruths = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_1.mat')[np.arange(99,245,5),:,:,:]                    

        training_iters = 700
        # training_iters = 50

    if mode == 'train':
        data_provider = SimpleDataProvider(data, truths, data_cls = data_cls, data_cls_num=data_cls_num, process_dict = para_dict_use['proc_dict'], onehot_cls=True, verbose=False)
    elif mode == 'test' or mode == 'valid' or mode == 'test_on_train':
        data_provider = None 
    else:
        raise ValueError('Unknow mode' + mode)
    
    valid_provider = SimpleDataProvider(vdata, vtruths, data_cls = vdata_cls, data_cls_num=data_cls_num, process_dict = para_dict_use['proc_dict'], onehot_cls=True, verbose=False)
    return data_provider, valid_provider, data_channels, truth_channels, training_iters
