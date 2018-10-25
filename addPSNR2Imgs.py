#%% Functions and packages
from PIL import Image, ImageDraw, ImageFont
import numpy as np
def notePSNR2Imgs(cleanImgs, noisyImgs, img_num = None):
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

def computePSNR(clearImg, noisyImg):
    # Format convertion
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

#%% 
# Set directories
import os, pathlib
resultPath = '../result'
for gpu in os.listdir(resultPath):    
    for expFolder in os.listdir(resultPath + '/' + gpu):
        if 'validation_noted' in os.listdir(resultPath + '/' + gpu + '/' + expFolder):
#        if False:
            # if the folder already procceed, just skip it
            continue
        else:
            print('PROCESSING: ' + gpu + '/' + expFolder)
            # Create directories
            valPath = resultPath + '/' + gpu + '/' + expFolder + '/validation/'
            valNotedPath = resultPath + '/' + gpu + '/' + expFolder +  '/validation_noted/'
            traNotedPath = resultPath + '/' + gpu + '/' + expFolder +  '/training_noted/'
            pathlib.Path(valNotedPath).mkdir(parents=True, exist_ok=True) 
            pathlib.Path(traNotedPath).mkdir(parents=True, exist_ok=True) 
            
            # Read valid input and target
            valid_inputs_file = valPath + 'trainOb_img.tif';  valNoisyImgs = Image.open(valid_inputs_file)
            valid_targets_file = valPath + 'trainGt_img.tif'; valCleanImgs = Image.open(valid_targets_file)
            noisyImgs_noted = notePSNR2Imgs(valCleanImgs, valNoisyImgs)
            noisyImgs_noted.save(valNotedPath + 'trainOb_img.png')
            
            # Proccessing files
            for file in os.listdir(valPath):
                if file.endswith('.tif'):
                    paras = file.split('_')
                    if 'train' in paras and 'outputs' in paras:
                        # Get file names
                        _, epoch, _, typ, _ = paras
                        train_inputs_file = valPath + 'epoch_{}_train_inputs_img.tif'.format(epoch)
                        train_outputs_file = valPath + 'epoch_{}_train_outputs_img.tif'.format(epoch)
                        train_targets_file = valPath + 'epoch_{}_train_targets_img.tif'.format(epoch)
                        
                        # Read images
                        noisyImgs = Image.open(train_inputs_file)
                        reconImgs = Image.open(train_outputs_file)
                        cleanImgs = Image.open(train_targets_file)
                        
                        # Note PSNR
                        reconImgs_noted = notePSNR2Imgs(cleanImgs, reconImgs)
                        noisyImgs_noted = notePSNR2Imgs(cleanImgs, noisyImgs)
                        
                        # Save noted images
                        noisyImgs_noted.save(traNotedPath + 'epoch_{}_train_inputs_img_noted.png'.format(epoch))
                        reconImgs_noted.save(traNotedPath + 'epoch_{}_train_outputs_img_noted.png'.format(epoch))
                        cleanImgs.save(traNotedPath + 'epoch_{}_train_targets_img_noted.png'.format(epoch))            
                        
                    elif 'valid' in paras:
                        _, epoch, typ, _ = paras                
                        valid_outputs_file = valPath + file
                        reconImgs = Image.open(valid_outputs_file)
                        reconImgs_noted = notePSNR2Imgs(valCleanImgs, reconImgs)
                        reconImgs_noted.save(valNotedPath + 'epoch_{}_valid_img.png'.format(epoch))
                else:
                    continue
                