import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint    

def crop_img(img, crop_boxes, saveMode='single'):
    if len(np.shape(img)) == 3:
        return crop_img_color(img, crop_boxes, saveMode)
    else:
        return crop_img_gray(img, crop_boxes, saveMode)

def crop_img_gray(img, crop_boxes, saveMode='single'):
    """
    Args:
        img: single img (h,w,c) or (h,w)
        crop_boxes: [(up, down, left, right)]
    """
    crop_num = len(crop_boxes)
    crop_h = crop_boxes[0][1] - crop_boxes[0][0]
    crop_w = crop_boxes[0][3] - crop_boxes[0][2]
        
    if saveMode == 'single':
        crops = []
    elif saveMode == 'Concat':
        col_num = int(np.floor(np.sqrt(crop_num)))
        row_num = int(np.ceil(np.sqrt(crop_num)))
        crops = np.zeros((row_num*crop_h,col_num*crop_w))
            
    elif saveMode == 'npy':
        crops = np.zeros((crop_num, crop_h, crop_w))
    
    for cropIdx in range(crop_num):        
        croped = img[crop_boxes[cropIdx][0]:crop_boxes[cropIdx][1], crop_boxes[cropIdx][2]:crop_boxes[cropIdx][3]]        
            
        if saveMode == 'Concat':
            col = cropIdx%col_num
            row = int(cropIdx/col_num)
            crops[row*crop_h: row*crop_h+ crop_h, col*crop_w: col*crop_w+ crop_w] = croped
        elif saveMode == 'single':
            crops.append(croped)
        elif saveMode == 'npy':
            crops[cropIdx,:,:] = croped
    
    return crops    

def crop_img_color(img, crop_boxes, saveMode='single'):
    """
    Args:
        img: single img (h,w,c) or (h,w)
        crop_boxes: [(up, down, left, right)]
    """
    crop_num = len(crop_boxes)
    crop_h = crop_boxes[0][1] - crop_boxes[0][0]
    crop_w = crop_boxes[0][3] - crop_boxes[0][2]
    img_c = img.shape[-1]
        
    if saveMode == 'single':
        crops = []
    elif saveMode == 'Concat':
        col_num = int(np.floor(np.sqrt(crop_num)))
        row_num = int(np.ceil(np.sqrt(crop_num)))
        crops = np.zeros((row_num*crop_h,col_num*crop_w, img_c))
            
    elif saveMode == 'npy':        
        crops = np.zeros((crop_num, crop_h, crop_w, img_c))
    
    for cropIdx in range(crop_num):        
        croped = img[crop_boxes[cropIdx][0]:crop_boxes[cropIdx][1], crop_boxes[cropIdx][2]:crop_boxes[cropIdx][3], :]
            
        if saveMode == 'Concat':
            col = cropIdx%col_num
            row = int(cropIdx/col_num)
            crops[row*crop_h: row*crop_h+ crop_h, col*crop_w: col*crop_w+ crop_w, :] = croped
        elif saveMode == 'single':
            crops.append(croped)
        elif saveMode == 'npy':
            crops[cropIdx,:,:,:] = croped
    
    return crops

def save_crops(crops, savePath=None, saveMode = 'Concat', saveStartIdx=0, energy_low_thres = 0, prefix=''):
    if savePath == None:
        raise ValueError('Please assign save path!')
        
    if saveMode == 'single':
        crop_num = len(crops)
        for cropIdx in range(crop_num):
            if np.sum(crops[cropIdx]) > energy_low_thres:
                if prefix != '':
                    plt.imsave(savePath+'/%s-%s.png' % (prefix,str(cropIdx + saveStartIdx)), crops[cropIdx], cmap='gray')
                else:
                    plt.imsave(savePath+'/%s.png' % prefix,str(cropIdx + saveStartIdx), crops[cropIdx], cmap='gray')
    elif saveMode == 'Concat':
#        crop_num = np.shape(crops)[0]
        crop_h = np.shape(crops[0])[0]
        if prefix != '':
            plt.imsave(savePath+'/{}_concat_{}_{}.png'.format(prefix, crop_h, saveStartIdx), crops)
        else:
            plt.imsave(savePath+'/concat_{}_{}.png'.format(crop_h, saveStartIdx), crops)
    elif saveMode == 'npy':
        crop_h = np.shape(crops)[1]
        np.save(savePath+'/patches_{}_{}'.format(crop_h, saveStartIdx), crops)
    

def grid_crop(img, crop_size = (30,30),
              save = False, savePath=None, saveMode = 'Concat', saveStartIdx=0):
    h, w = np.shape(img)[0:2]
    crop_h, crop_w = crop_size[:]
    col_num = int(w/crop_size[1])
    row_num = int(h/crop_size[0])
    crop_boxes = []
    for rowIdx in range(row_num):
        for colIdx in range(col_num):
            crop_boxes.append((rowIdx * crop_h, (rowIdx+1) * crop_h,
                               colIdx * crop_w, (colIdx+1) * crop_w))
    return crop_img(img, crop_boxes, saveMode), crop_boxes

def random_crop(img = None, crop_num = 20, crop_size = (10,10), 
                energy_low_thres = 0,
                save = False, savePath=None, saveMode = 'Concat', saveStartIdx=0, randomMode = 'uniform'):
    img_min= np.min(img)
    img_max = np.max(img) - img_min
    img = (img-img_min)/img_max
    # crop_size: (patch height, patch width)
    crop_h = crop_size[0]
    crop_h_up = np.int(np.floor((crop_h - 1) / 2))
    crop_h_down = np.int(np.ceil((crop_h - 1) / 2))
    
    crop_w = crop_size[1]
    crop_w_left = np.int(np.floor((crop_w - 1) / 2))
    crop_w_right = np.int(np.ceil((crop_w - 1) / 2))
    
    if len(np.shape(img))>2:
        img_h, img_w, img_c = np.shape(img)
    else:
        img_h, img_w = np.shape(img)
    crop_h_upperBound = crop_h_up
    crop_h_lowerBound = img_h - crop_h_down
    crop_w_leftBound = crop_w_left
    crop_w_rightBound = img_w - crop_w_right
    
    if randomMode == 'uniform':
        center_ys = randint(crop_h_upperBound, crop_h_lowerBound, size = crop_num)
        center_xs = randint(crop_w_leftBound, crop_w_rightBound, size = crop_num)
        crop_boxes = []
        for cropIdx in range(crop_num):
            crop_boxes.append((center_ys[cropIdx] - crop_h_up, center_ys[cropIdx] + crop_h_down +1,
                                   center_xs[cropIdx] - crop_w_left, center_xs[cropIdx] + crop_w_right +1))
    elif randomMode == 'normal' or randomMode == 'Gaussian':
        pass
    else:
        raise ValueError('Unknown randomMode %s' % randomMode)
        
    crops = crop_img(img, crop_boxes, saveMode)
    if save:
        save_crops(crops, savePath, saveMode, saveStartIdx, energy_low_thres)    
    return crops, crop_boxes
