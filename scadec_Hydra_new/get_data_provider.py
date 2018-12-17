from scadec_Hydra_new.image_util_new import SimpleDataProvider

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
                    # if ('1000' in para_dict_use['Ob']):
                    #     if mode == 'train':
                    #         data = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_1000_neigh_motion_part_2.mat')
                    #         data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]),n_classes = data_cls_num, mode='equally')
                    #         #data_cls = idx_classify(np.arange(100, 245+1, 5), mode='equally_960')
                    #     elif mode == 'test' or mode == 'valid':
                    #         data = None
                    #     vdata = h5py_mat2npy('../data/valid_np/valOb_FULL_SEG_1000_neigh_motion.mat')
                    # else:
                    #     if mode == 'train':
                    #         data = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
                    #         data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]),n_classes = data_cls_num, mode='equally')
                    #         #data_cls = idx_classify(np.arange(100, 245+1, 5), mode='equally_960')
                    #     elif mode == 'test' or mode == 'valid':
                    #         data = None
                    #     vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion.mat')
                    if para_dict_use.get('ObSampleLineNum', False):
                        sampleLineNum = para_dict_use.get('ObSampleLineNum', False)
                        if mode == 'train':
                            data = h5py_mat2npy('../data/train_np/traOb' + '_T=' + str(sampleLineNum) + '_FULL_SEG_neigh_motion_part_2.mat')
                            data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]),n_classes = data_cls_num, mode='equally')
                        elif mode == 'test' or mode == 'valid':
                            data = None
                        vdata = h5py_mat2npy('../data/valid_np/valOb' + '_T=' + str(sampleLineNum) + '_FULL_SEG_neigh_motion.mat')
                    else:
                        pass

                    
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
                if para_dict_use.get('ObSampleLineNum', False):
                    # sampleLineNum = para_dict_use.get('ObSampleLineNum', False)
                    if mode == 'train':
                        truths = h5py_mat2npy('../data/train_np/traGt_T=2000_part_2.mat')
                        # truths = h5py_mat2npy('../data/valid_np/valGt.mat')
                        vtruths = h5py_mat2npy('../data/valid_np/valGt_T=2000.mat')
                    elif mode == 'test' or mode == 'valid':
                        truths = None
                        vtruths = h5py_mat2npy('../data/valid_np/valGt_T=2000.mat')
                else:
                    pass


        training_iters = 50
    else:
        # Obeservation / input noisy data
        if ('3C' in para_dict_use['Ob']):
            data_channels = 3 
            if 'motion' in para_dict_use['Ob']:
                if ('FULL_SEG' in para_dict_use['Ob']):
                    if mode == 'train':         
                        # if ('1000' in para_dict_use['Ob']):
                        # # data = h5py_mat2npy('train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
                        #     data1 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_1000_neigh_motion_part_1.mat')
                        #     data2 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_1000_neigh_motion_part_2.mat')
                        #     data3 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_1000_neigh_motion_part_3.mat')
                        #     data  = np.concatenate([data1, data2, data3], axis=0)    
                        #     vdata = h5py_mat2npy('../data/valid_np/valOb_FULL_SEG_1000_neigh_motion.mat')
                        # else:
                        #     data1 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')
                        #     data2 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_2.mat')
                        #     data3 = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_3.mat')
                        #     data  = np.concatenate([data1, data2, data3], axis=0)    
                        #     vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion.mat')
                        if para_dict_use.get('ObSampleLineNum', False):
                            sampleLineNum = para_dict_use.get('ObSampleLineNum', False)
                            if mode == 'train':
                                data1 = h5py_mat2npy('../data/train_np/traOb' + '_T=' + str(sampleLineNum) + '_FULL_SEG_neigh_motion_part_1.mat')
                                data2 = h5py_mat2npy('../data/train_np/traOb' + '_T=' + str(sampleLineNum) + '_FULL_SEG_neigh_motion_part_2.mat')
                                data3 = h5py_mat2npy('../data/train_np/traOb' + '_T=' + str(sampleLineNum) + '_FULL_SEG_neigh_motion_part_3.mat')
                                data  = np.concatenate([data1, data2, data3], axis=0)
                                data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]),n_classes = data_cls_num, mode='equally')
                                vdata = h5py_mat2npy('../data/valid_np/valOb' + '_T=' + str(sampleLineNum) + '_FULL_SEG_neigh_motion.mat')
                            elif mode == 'test' or mode == 'valid':
                                data = None
                            vdata = h5py_mat2npy('../data/valid_np/valOb' + '_T=' + str(sampleLineNum) + '_FULL_SEG_neigh_motion.mat')
                        else:
                            pass
                        data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]), n_classes = data_cls_num, mode='equally')
                        # print('Observation_size')
                        # print(np.shape(data)[0])
                        # print('data_cls (at image_utils.py): ')
                        # print(data_cls)
                        del(data1, data2, data3)
                        
                        vdata_idx = np.arange(100, 245+1, 5)%96
                        vdata_idx[vdata_idx==0] = 96
                        # vdata_cls = idx_classify(np.arange(100, 245+1, 5),n_classes = data_cls_num, mode='equally_96')    
                        vdata_cls = idx_classify(vdata_idx,n_classes = data_cls_num, mode='equally_96')    
                    elif mode == 'test' or mode == 'valid':
                        data = None
                        # if ('1000' in para_dict_use['Ob']):
                        #     vdata = h5py_mat2npy('../data/valid_np/valOb_neigh_motion_FULL.mat')
                        # else:
                        #     vdata = h5py_mat2npy('../data/valid_np/valOb_FULL_SEG_1000_neigh_motion.mat')
                        if para_dict_use.get('ObSampleLineNum', False):
                            vdata = h5py_mat2npy('../data/valid_np/valOb' + '_T=' + str(sampleLineNum) + '_FULL_SEG_neigh_motion_FULL.mat')
                        else:
                            pass
                        vdata_cls = idx_classify(assign_silce_idx(np.shape(vdata)[0]),n_classes = data_cls_num, mode = 'equally')
                        # vdata_cls = idx_classify(np.arange(100, 245+1, 5), mode='equally_960')
                    elif mode == 'test_on_train':
                        data=None
                        if ('1000' in para_dict_use['Ob']):
                            vdata = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')[np.arange(99,245,5),:,:,:]
                        else:
                            vdata = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_1000_neigh_motion_part_1.mat')[np.arange(99,245,5),:,:,:]
                        # idx = np.arange(99,245,5)                        
                        vdata_idx = np.arange(100, 245+1, 5)%96
                        vdata_idx[vdata_idx==0] = 96
                        # vdata_cls = idx_classify(np.arange(100, 245+1, 5),n_classes = data_cls_num, mode='equally_96')    
                        vdata_cls = idx_classify(vdata_idx,n_classes = data_cls_num, mode='equally_96')    
                        # vdata_cls = idx_classify(assign_silce_idx(np.shape(vdata)[0]),n_classes = data_cls_num, mode = 'equally')
        else:
            data_channels = 3 
            if mode == 'train':                        
                if para_dict_use.get('ObSampleLineNum', False):
                    sampleLineNum = para_dict_use.get('ObSampleLineNum', False)
                    if mode == 'train':
                        data1 = h5py_mat2npy('../data/train_np/traOb' + '_T=' + str(sampleLineNum) + '_part_1.mat')
                        data2 = h5py_mat2npy('../data/train_np/traOb' + '_T=' + str(sampleLineNum) + '_part_2.mat')
                        data3 = h5py_mat2npy('../data/train_np/traOb' + '_T=' + str(sampleLineNum) + '_part_3.mat')
                        data  = np.concatenate([data1, data2, data3], axis=0)
                        data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]),n_classes = data_cls_num, mode='equally')
                        vdata = h5py_mat2npy('../data/valid_np/valOb' + '_T=' + str(sampleLineNum) + '.mat')
                    elif mode == 'test' or mode == 'valid':
                        data = None
                    vdata = h5py_mat2npy('../data/valid_np/valOb' + '_T=' + str(sampleLineNum) + '.mat')
                else:
                    pass
                data_cls = idx_classify(assign_silce_idx(np.shape(data)[0]), n_classes = data_cls_num, mode='equally')            
                del(data1, data2, data3)
                
                vdata_idx = np.arange(100, 245+1, 5)%96
                vdata_idx[vdata_idx==0] = 96
                vdata_cls = idx_classify(vdata_idx,n_classes = data_cls_num, mode='equally_96')    
            elif mode == 'test' or mode == 'valid':
                data = None
                if para_dict_use.get('ObSampleLineNum', False):
                    vdata = h5py_mat2npy('../data/valid_np/valOb' + '_T=' + str(sampleLineNum) + '.mat')
                else:
                    pass
                vdata_cls = idx_classify(assign_silce_idx(np.shape(vdata)[0]),n_classes = data_cls_num, mode = 'equally')
                # vdata_cls = idx_classify(np.arange(100, 245+1, 5), mode='equally_960')
            elif mode == 'test_on_train':
                data=None
                if ('1000' in para_dict_use['Ob']):
                    vdata = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_neigh_motion_part_1.mat')[np.arange(99,245,5),:,:,:]
                else:
                    vdata = h5py_mat2npy('../data/train_np/traOb_FULL_SEG_1000_neigh_motion_part_1.mat')[np.arange(99,245,5),:,:,:]
                # idx = np.arange(99,245,5)                        
                vdata_idx = np.arange(100, 245+1, 5)%96
                vdata_idx[vdata_idx==0] = 96
                # vdata_cls = idx_classify(np.arange(100, 245+1, 5),n_classes = data_cls_num, mode='equally_96')    
                vdata_cls = idx_classify(vdata_idx,n_classes = data_cls_num, mode='equally_96')    



        # Ground truth / Target clean data
        if ('3C' in para_dict_use['Gt']):
            truth_channels = 3
            pass
        else:
            truth_channels = 1
            if ('FULL_SEG' in para_dict_use['Gt']):
                if para_dict_use.get('ObSampleLineNum', False):
                    if mode == 'train':
                        # truths = h5py_mat2npy('train_np/traGt_FULL_SEG_part_1.mat')
                        # truths1 = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_1.mat')
                        # truths2 = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_2.mat')
                        # truths3 = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_3.mat')
                        truths1 = h5py_mat2npy('../data/train_np/traGt_T=2000_part_1.mat')
                        truths2 = h5py_mat2npy('../data/train_np/traGt_T=2000_part_2.mat')
                        truths3 = h5py_mat2npy('../data/train_np/traGt_T=2000_part_3.mat')
                        truths  = np.concatenate([truths1, truths2, truths3], axis=0)
                        # del(truths1, truths2, truths3)                
                        vtruths = h5py_mat2npy('../data/valid_np/valGt_T=2000.mat')
                    elif mode == 'test' or mode == 'valid':
                        truths = None
                        vtruths = h5py_mat2npy('../data/valid_np/valGt_T=2000_FULL.mat')
                    elif mode == 'test_on_train':
                        vtruths = h5py_mat2npy('../data/train_np/traGt_FULL_SEG_part_1.mat')[np.arange(99,245,5),:,:,:]                    
                else:
                    pass

        training_iters = 700
        # training_iters = 50
    ignore_classes = para_dict_use.get('ignore_classes', [])
    # def find_min_empty_cls(cls_array, startCls = 0):
    #     currentCls = startCls
    #     while np.sum(cls_array[cls_array==currentCls]) != 0:
    #         currentCls += 1
    #     return currentCls

    # print(vdata_cls)
    mask_time = {'year':2018,
           'month':11,
           'day':9,
           'hour':10}
    maskMode = 'load' # 'new', 'load'
    # maskMode = 'new' # 'new', 'load'
    if para_dict_use.get('preMask', False):
        model_path = './scadec_Hydra/maskCNN/CNN_mask-{}-{}-{}-{}-2000.meta'.format(mask_time['year'], mask_time['month'], mask_time['day'], mask_time['hour'])
        # Getting vmasks
        print('Loading vmasks...')
        # vdata_masks = np.load('../data/masks/vdata_FULL_SEG_28_28_masks.npy')        
        if DEBUG_MODE:
            saveName = 'vdata_'+ para_dict_use['Gt']+'_{}_{}_{}_{}'.format(mask_time['year'],mask_time['month'],mask_time['day'],mask_time['hour'])
            save_path = '../data/masks_debug/'
            if maskMode == 'new':                
                vdata_masks = predict_masks(vtruths, savePatches=False, saveYPre=True, saveMasks=True, save_path = save_path,saveName=saveName, model_path=model_path)
                # vdata_masks = predict_masks(vtruths, savePatches=False, loadYPre=True, saveMasks=True, save_path = save_path,saveName=saveName, model_path=model_path)
            else:
                vdata_masks = np.load(save_path + saveName+'_28_28_masks.npy')
        else:
            saveName = 'vdata_'+ para_dict_use['Gt']+'_{}_{}_{}_{}'.format(mask_time['year'],mask_time['month'],mask_time['day'],mask_time['hour'])
            save_path = '../data/masks/'
            if maskMode == 'new':                
                vdata_masks = predict_masks(vtruths, savePatches=False, saveYPre=True, saveMasks=True, save_path = save_path, saveName=saveName, model_path=model_path)
                # vdata_masks = predict_masks(vtruths, savePatches=False, loadYPre=True, saveMasks=True, save_path = save_path, saveName=saveName, model_path=model_path)
            else:
                vdata_masks = np.load(save_path + saveName+'_28_28_masks.npy')
            
            
        # vdata_masks = predict_masks(vtruths, savePatches=False, loadYPre=True, saveMasks=True, saveName='vdata_'+ para_dict_use['Gt'])
        # Getting data masks
        if data is not None:            
            saveName = 'data_'+ para_dict_use['Gt']+'_{}_{}_{}_{}'.format(mask_time['year'],mask_time['month'],mask_time['day'],mask_time['hour'])
            if maskMode == 'new':
                print('Computing Masks...')
                if DEBUG_MODE:
                    save_path = '../data/masks_debug/'
                    data_masks = predict_masks(truths, savePatches=False, saveYPre=True, saveMasks=True, save_path=save_path,saveName=saveName, model_path=model_path)
                    # data_masks = predict_masks(truths, savePatches=False, loadYPre=True, saveMasks=True, save_path=save_path,saveName=saveName, model_path=model_path)
                else:
                    save_path = '../data/masks/'                
                    data_masks_1 = predict_masks(truths1, savePatches=False, saveYPre=True, saveMasks=True, save_path = save_path, saveName=saveName + '_1', model_path=model_path)
                    data_masks_2 = predict_masks(truths2, savePatches=False, saveYPre=True, saveMasks=True, save_path = save_path, saveName=saveName + '_2', model_path=model_path)
                    data_masks_3 = predict_masks(truths3, savePatches=False, saveYPre=True, saveMasks=True, save_path = save_path, saveName=saveName + '_3', model_path=model_path)
                    data_masks  = np.concatenate([data_masks_1, data_masks_2, data_masks_3], axis=0)
            
            else:
            # data_masks = predict_masks(truths, savePatches=False, saveYPre=True, saveMasks=True, saveName='data_'+ para_dict_use['Gt'])
                print('Loading masks...')
                if DEBUG_MODE:
                    save_path = '../data/masks_debug/'
                    data_masks = np.load(save_path+saveName+'_28_28_masks.npy')
                else:
                    save_path = '../data/masks/'
                    data_masks_1 = np.load(save_path+saveName+ '_1' +'_28_28_masks.npy')
                    data_masks_2 = np.load(save_path+saveName+ '_2' +'_28_28_masks.npy')
                    data_masks_3 = np.load(save_path+saveName+ '_3' +'_28_28_masks.npy')
                    data_masks  = np.concatenate([data_masks_1, data_masks_2, data_masks_3], axis=0)    

            # data_masks = predict_masks(truths, savePatches=False, loadYPre=True, saveMasks=True, saveName='data_'+ para_dict_use['Gt'])
        else:
            data_masks = None    
    else:
        data_masks = None
        vdata_masks = None

    for idx, ignore_class in enumerate(ignore_classes):
        ignore_class -= idx
        data = data[data_cls != ignore_class,:,:,:]
        vdata = vdata[vdata_cls != ignore_class,:,:,:]        
        truths = truths[data_cls != ignore_class,:,:,:]
        vtruths = vtruths[vdata_cls != ignore_class,:,:,:]
        if para_dict_use.get('preMask', False):
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

    # Add
    if data_masks is not None:
        print('Adding Masks...')
        data_masks += para_dict_use.get('preMaskAdd', 0)
    if vdata_masks is not None:
        print('Adding VMasks...')
        vdata_masks += para_dict_use.get('preMaskAdd', 0)
    
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
from scipy.signal import convolve2d

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
    # print(np.shape(masks))
    
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
            y_pre_raws = np.load(save_path + '{}_{}_{}_ypre_raws.npy'.format(saveName, cropSize[0], cropSize[1]))
            boxeses = np.load(save_path + '{}_{}_{}_boxeses.npy'.format(saveName, cropSize[0], cropSize[1]))
        elif not loadMaskes:
            y_pre_raws = []
            for imgIdx in tqdm(range(img_n),ncols=75):
                y_pre_raws.append(sess.run(pred, feed_dict={xs:patches[imgIdx*crop_num:(imgIdx+1)*crop_num]}))
        
            # a = 1
        if saveYPre:
            np.save(save_path + '{}_{}_{}_ypre_raws.npy'.format(saveName, cropSize[0], cropSize[1]), y_pre_raws)
            np.save(save_path + '{}_{}_{}_boxeses.npy'.format(saveName, cropSize[0], cropSize[1]), boxeses)
        
        # Get masks
        # y_pres = []
        print('Computing masks...')
        fMean = np.ones((5,5))/25
        masks = np.zeros(np.shape(test_imgs))
        for imgIdx in tqdm(range(img_n),ncols=75):
            # y_pre = y_pre_raws[imgIdx][:,0]>0.9
            boxes = boxeses[imgIdx]
            mask = np.zeros(np.shape(test_imgs[0,:,:,0]))
            for idx, box in enumerate(boxes):
                mask[box[0]:box[1],box[2]:box[3]] += 0.01*y_pre_raws[imgIdx][idx, 0]
                # if y_pre[idx] == True:
                #     mask[box[0]:box[1],box[2]:box[3]] += 0.01
            mask = convolve2d(mask, fMean, mode='same')
            masks[imgIdx] = np.reshape(mask, [img_h, img_w, img_c])
            
        if saveMasks:
            np.save(save_path + '{}_{}_{}_masks'.format(saveName, cropSize[0], cropSize[1]), masks)
        endTime = time.time()
        print('Finished computing masks for {} within {} mins for {} images!'.format(saveName, (endTime - startTime)/60, img_n))
    return masks