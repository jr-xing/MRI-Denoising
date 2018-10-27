import numpy as np
import os

#%% Load Configuration
from configs_Hrdra import para_dict_use_train, para_str_use_train
import pprint
pprint.pprint('Running training: '+ para_str_use_train)
pprint.pprint(para_dict_use_train)

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = para_dict_use_train.get('GPU_IND', '3')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind # 0,1,2,3

#%% Load data
from scadec_Hydra.image_util import get_data_provider
data_provider, valid_provider, data_channels, truth_channels, training_iters = get_data_provider(para_dict_use_train, 'train', DEBUG_MODE=False)

# -----------------------------------Loss------------------------------------------------------- #
losses_dict = para_dict_use_train['losses']
kwargs = para_dict_use_train['kwargs']

####################################################
####                  NETWORK                    ###
####################################################

"""
	here we specify the neural network.

"""
from scadec_Hydra.unet_bn_Hydra import Unet_bn
net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost_dict_list=losses_dict, **kwargs)

####################################################
####                 TRAINING                    ###
####################################################
from scadec_Hydra.train_Hydra import Trainer_bn

# args for training
batch_size = kwargs.get("batch_size", 5) # batch size for training
valid_size = kwargs.get("valid_size", 5)  # batch size for validating
optimizer = "adam"  # optimizer we want to use, 'adam' or 'momentum'

# # output paths for results
output_path = '../result/gpu' + gpu_ind + '/' + para_str_use_train + '/models'
prediction_path = '../result/gpu' + gpu_ind + '/' + para_str_use_train + '/validation'

# # optional args
opt_kwargs = {
		'learning_rate': 0.0001#EDIT
}

# # make a trainer for scadec
# # epochs=600
epochs=para_dict_use_train.get('epochs', 200)
import time
time_start= time.time()
trainer = Trainer_bn(net, batch_size=batch_size, optimizer = para_dict_use_train.get('optimizer','adam'), opt_kwargs=opt_kwargs, verbose=False)
path = trainer.train(data_provider, output_path, valid_provider, valid_size, dropout=para_dict_use_train['Keep'], training_iters=training_iters, epochs=epochs, display_step=100, save_epoch=20, prediction_path=prediction_path)
time_end = time.time()
print(time_end - time_start)
