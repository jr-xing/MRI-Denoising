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
    
    def __init__(self, data, truths, data_cls = None, data_cls_num=None, onehot_cls = False, process_dict = {}):
        # additional_info should be list of dicts
        super(SimpleDataProvider, self).__init__()
        # Xing
        self.data = data
        self.truths = truths
        #self.data = np.float64(data)
        #self.truths = np.float64(truths)
        self.img_channels = self.data[0].shape[2]
        self.truth_channels = self.truths[0].shape[2]
        self.file_count = data.shape[0]
        self.process_dict = process_dict
        
        # data_cls should be np.array
        self.data_cls = data_cls.astype(np.int32)
        if data_cls_num==None:
            data_cls_num = len(np.unique(self.data_cls))
        else:
            self.data_cls_num = data_cls_num
        
        self.onehot_cls = onehot_cls


    def _next_batch(self, n):
        idx = np.random.choice(self.file_count, n, replace=False)
        img = self.data[idx[0]]
        nx = img.shape[0]
        ny = img.shape[1]
        X = np.zeros((n, nx, ny, self.img_channels))
        Y = np.zeros((n, nx, ny, self.truth_channels))

        if self.onehot_cls:
            batch_cls = np.zeros([n, self.data_cls_num])
            # print('batch_cls created')
            # print(batch_cls)
        else:
            batch_cls = np.zeros(n)

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

    