import numpy as np
import os

#%% Load Configuration
from configs_Hrdra import para_dict_use_test, para_str_use_test
import pprint
pprint.pprint('Running test: '+ para_str_use_test)
pprint.pprint(para_dict_use_test)

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = para_dict_use_test.get('GPU_IND', '3')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind # 0,1,2,3
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 0,1,2,3

#%% Load data
from scadec_Hydra.image_util import get_data_provider
DEBUG_MODE = False
NOTE_PSNR = False
data_provider, valid_provider, data_channels, truth_channels, training_iters = get_data_provider(para_dict_use_test, 'test', DEBUG_MODE=DEBUG_MODE)

# -----------------------------------Loss------------------------------------------------------- #
losses_dict = para_dict_use_test['losses']
kwargs = para_dict_use_test['kwargs']

####################################################
####                  NETWORK                    ###
####################################################

"""
	here we specify the neural network.

"""
from scadec_Hydra.unet_bn_Hydra import Unet_bn
net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost_dict_list=losses_dict, **kwargs)

####################################################
####                  Test                       ###
####################################################

num = valid_provider.size
predicts = []
valid_x, valid_y, batch_cls = valid_provider('full')
model_path =  '../result/gpu' + str(gpu_ind) + '/' + para_str_use_test + '/models/final/model.cpkt'
if DEBUG_MODE:
    test_save_path = '../result/gpu' + str(gpu_ind) + '/' + para_str_use_test + '/test_small'
else:
    test_save_path = '../result/gpu' + str(gpu_ind) + '/' + para_str_use_test + '/test'

import pathlib
pathlib.Path(test_save_path).mkdir(parents=True, exist_ok=True) 

test_batch_size = 5
test_batch_num = int(num/test_batch_size)

# Average psnr for each class
n_cls = kwargs.get('n_classes', 1)
avg_psnr_cls = [0]*n_cls

from scadec_Hydra import util
for batchIdx in range(test_batch_num):

    print('')
    # print('')
    print('************* {}/{} *************'.format(batchIdx+1, test_batch_num))
    print('')
    # print('')
    imgIdxStart = batchIdx*test_batch_size
    imgIdxEnd   = (batchIdx+1)*test_batch_size
    x = valid_x[imgIdxStart:imgIdxEnd,:,:,:]
    y = valid_y[imgIdxStart:imgIdxEnd,:,:,:]
    x_cls = batch_cls[imgIdxStart:imgIdxEnd,:]
    x_cls_arr = [clas for clas in list(x_cls.argmax(1))]
    x_cls_str = ['cls: '+ str(clas) for clas in list(x_cls.argmax(1))]

    predict = net.predict(model_path = model_path, x_test = x, batch_cls = x_cls, keep_prob = 1, phase=False)
    predicts.append(predict[0:test_batch_size,:,:])
    psnrs = util.computePSNRs(y, predict)
    for clasIdx, clas in enumerate(x_cls_arr):
        avg_psnr_cls[clas] += psnrs[clasIdx]
    #total_avg_psnr += avg_psnr
    test_inputs = util.concat_n_images(x)    
    test_targets = util.concat_n_images(y)
    test_outputs = util.concat_n_images(predict)
    if NOTE_PSNR:        
        batch_inputs_psnr_str = ['PSNR: %.3f'%psnr  for psnr in list(util.computePSNRs(test_targets, test_inputs))]
        batch_outputs_psnr_str = ['PSNR: %.3f'%psnr for psnr in list(util.computePSNRs(test_targets, test_outputs))]
        batch_inputs_str = list(zip(batch_inputs_psnr_str, x_cls_str))
        batch_outputs_str = list(zip(batch_outputs_psnr_str, x_cls_str))
        
        test_inputs = util.noteTexts2Imgs(test_inputs, batch_inputs_str)
        test_outputs = util.noteTexts2Imgs(test_outputs, batch_outputs_str)

    # util.save_img(img, "%s/%s_img.tif"%(test_save_path, name))
    util.save_img(test_inputs, "{}/batch_{}_inputs_img.png".format(test_save_path, batchIdx))
    util.save_img(test_outputs, "{}/batch_{}_outputs_img.png".format(test_save_path, batchIdx))
    util.save_img(test_targets, "{}/batch_{}_targets_img.png".format(test_save_path, batchIdx))

predicts = np.concatenate(predicts, axis=0)
util.save_mat(predicts, test_save_path+'/test.mat')

for idx in range(len(avg_psnr_cls)):
    avg_psnr_cls[idx] = avg_psnr_cls[idx] / (96/n_cls)
print('PSNR for each class:')
print(avg_psnr_cls)

file = open("{}/avg_psnr.txt".format(test_save_path),"w") 
file.write("PSNR for each class:")
file.write(str(avg_psnr_cls))
file.close()
# file.write(“Hello World”) 
# file.write(“This is our new text file”) 
# file.write(“and this is another line.”) 
# file.write(“Why? Because we can.”) 
 
# file.close() 

#avg_psnr = total_avg_psnr / test_batch_size
#print('avg_psnr %f' % avg_psnr)



# from scadec_Hydra.train_Hydra import Trainer_bn

# # args for training
# batch_size = kwargs.get("batch_size", 5) # batch size for training
# valid_size = kwargs.get("valid_size", 5)  # batch size for validating
# optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

# # # output paths for results
# output_path = '../result/gpu' + gpu_ind + '/' + para_dict_use_test + '/models'
# prediction_path = '../result/gpu' + gpu_ind + '/' + para_dict_use_test + '/validation'

# # # optional args
# opt_kwargs = {
# 		'learning_rate': 0.0001#EDIT
# }

# # # make a trainer for scadec
# # # epochs=600
# epochs=para_dict_use_test.get('epochs', 200)
# import time
# time_start= time.time()
# trainer = Trainer_bn(net, batch_size=batch_size, optimizer = para_dict_use_test.get('optimizer','adam'), opt_kwargs=opt_kwargs)
# path = trainer.train(data_provider, output_path, valid_provider, valid_size, dropout=para_dict_use_test['Keep'], training_iters=training_iters, epochs=epochs, display_step=100, save_epoch=20, prediction_path=prediction_path)
# time_end = time.time()
# print(time_end - time_start)
