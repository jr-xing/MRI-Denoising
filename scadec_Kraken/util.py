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
import numpy as np
import scipy.io as sio
import scipy.misc as smisc


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img = img/np.amax(img)
    img *= 255
    return img

def to_double(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    return img

def save_mat(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    
    sio.savemat(path, {'img':img})


def save_img(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    img = to_rgb(img)
    smisc.imsave(path, img.round().astype(np.uint8))

# Xing
from PIL import Image, ImageDraw, ImageFont
import numpy as np
def noteTexts2Imgs(imgs, textListList, img_num = None):
    # imgs is a long image
    # Get single image size
    if type(imgs) == np.ndarray:
        if np.max(imgs) > 2:
            imgs = Image.fromarray(np.uint8(imgs))
        else:
            img_min = np.min(imgs)
            img_max = np.max(imgs)
            imgs = (imgs - img_min)/(img_max-img_min)
            imgs = Image.fromarray(np.uint8(imgs*255))

    H = imgs.size[1]
    Ws = imgs.size[0]
    if img_num == None:
        img_num = int(Ws/H)
        W = H
    else:
        W = int(Ws/img_num)
    notedImg = Image.new('RGB', (Ws,H))

    # print(textListList)
    for imgIdx in range(img_num):
        # crop(box), box –  The crop rectangle, as a (left, upper, right, lower)-tuple.
        imgLocation = (imgIdx*W, 0, (imgIdx+1)*W, H)
        img = imgs.crop(imgLocation)      
        
        # print('imgIdx {}'.format(imgIdx))
        notedSingleImg = noteSingleImg(img, textListList[imgIdx], fontSize = 12)
        notedImg.paste(notedSingleImg, imgLocation)
    return np.array(notedImg)

def notePSNR2Imgs(cleanImgs, noisyImgs, img_num = None):
    # cleanImgs: a wide image containing many images
    # Get single image size
    H = cleanImgs.size[1]
    Ws = cleanImgs.size[0]
    if img_num == None:
        img_num = int(Ws/H)
        W = H
    else:
        W = int(Ws/img_num)
    notedImg = Image.new('RGB', (Ws,H))
    for imgIdx in range(img_num):
        # crop(box), box –  The crop rectangle, as a (left, upper, right, lower)-tuple.
        imgLocation = (imgIdx*W, 0, (imgIdx+1)*W, H)
        cleanImg = cleanImgs.crop(imgLocation)
        noisyImg = noisyImgs.crop(imgLocation)
        psnr = computePSNR(cleanImg, noisyImg)
        # notedNoisyImg = noteSingleImg(noisyImg, ['PSNR: '+ str(psnr), 'CLS: ' + str(1)], fontSize = 12)
        notedNoisyImg = noteSingleImg(noisyImg, ['PSNR: %.3f' % psnr], fontSize = 12)
        notedImg.paste(notedNoisyImg, imgLocation)
    return notedImg

def computePSNRs(cleanImgs, noisyImgs, img_num = None):
    # cleanImgs: a wide image containing many images, should be numpy array
    # Get single image size
    if type(cleanImgs) == Image.Image:
        cleanImgs = np.array(cleanImgs)
        noisyImgs = np.array(noisyImgs)

    H = cleanImgs.shape[0]
    Ws = cleanImgs.shape[1]
    if img_num == None:
        img_num = int(Ws/H)
        W = H
    else:
        W = int(Ws/img_num)
    
    psnrs = []
    for imgIdx in range(img_num):
        # crop(box), box –  The crop rectangle, as a (left, upper, right, lower)-tuple.
        #imgLocation = (imgIdx*W, 0, (imgIdx+1)*W, H)
        #cleanImg = cleanImgs.crop(imgLocation)
        #noisyImg = noisyImgs.crop(imgLocation)
        cleanImg = cleanImgs[:,imgIdx*W:(imgIdx+1)*W]
        noisyImg = noisyImgs[:,imgIdx*W:(imgIdx+1)*W]
        psnr = computePSNR(cleanImg, noisyImg)
        psnrs.append(psnr)
    return psnrs

def computePSNR(clearImg, noisyImg):
    # Format convertion
    if type(clearImg) == Image.Image:
        clearImg = np.array(clearImg)
        noisyImg = np.array(noisyImg)

    if len(clearImg.shape) == 3:
        clearImg = clearImg[:,:,1]
    if len(noisyImg.shape) == 3:
        noisyImg = noisyImg[:,:,1]    
        
    # MSE
    MSE = np.sum((clearImg-noisyImg)**2)/np.size(clearImg)
    
    # R
    if np.max(clearImg) > 2:
        # if image pixel value \in [0,255], R = 255
        R = 255
    else:
        # if image pixel value \in [0,1], R = 1
        R = 1
    
    # PSNR
    return 10*np.log10(R*R/MSE)

def noteSingleImg(img, texts, init_position = (10,10), fontSize = 16, lineSpace = 2):
    d = ImageDraw.Draw(img)
    
    for textIdx, text in enumerate(texts):
        d.text((init_position[0], init_position[1] + textIdx*fontSize + lineSpace ) , text, fill=(200,200,200), font=ImageFont.truetype('./DejaVuSans.ttf', fontSize))
    return img


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

def concat_n_images(image_mat, batch_cls = None):
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

# def debug_print(str):
#     global IFDEBUG
#     if IFDEBUG:
#         print(str)

def verbose_print(str, verbose):
    if verbose:
        print(str)

