####################################################
####                 FUNCTIONS                   ###
####################################################
import numpy as np
import os
import h5py

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