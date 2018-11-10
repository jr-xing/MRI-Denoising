import numpy as np
import os

#%% Load Configuration
from configs import para_dict_use_train, para_str_use_train
import pprint
pprint.pprint('Running training: '+ para_str_use_train)
pprint.pprint(para_dict_use_train)

# here indicating the GPU you want to use. if you don't have GPU, just leave it.
gpu_ind = para_dict_use_train.get('GPU_IND', '3')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind # 0,1,2,3

#%% Load data
from scadec_Kraken.image_util import get_data_provider
data_provider, valid_provider, data_channels, truth_channels, training_iters = get_data_provider(para_dict_use_train, 'train', DEBUG_MODE=False)

# -----------------------------------Loss------------------------------------------------------- #
from scadec_Kraken.unet_bn_Kraken import Unet_bn
losses_dict = para_dict_use_train['losses']
if type(para_dict_use_train['kwargs']) == dict:
	kwargs = para_dict_use_train['kwargs']
	kwargs_list = {}
	for key, value in kwargs.items():
		if key not in ['Ob','Gt','eopchs','optimizer','server','GPU_IND']:
			kwargs_list[key] = [value]

	# net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost_dict_list=losses_dict, **kwargs)

elif type(para_dict_use_train['kwargs']) == list:	
	kwargs_list = para_dict_use_train['kwargs']
	kwargs = kwargs_list[0]

# kwargs['structure']['n_classes'] -= len(para_dict_use_train.get('ignore_classes', []))
# kwargs['structure']['n_classes'] =  16 - len(para_dict_use_train.get('ignore_classes', []))
kwargs_list[0]['structure']['n_classes'] -= len(para_dict_use_train.get('ignore_classes', []))
net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost_dict_lists=losses_dict, kwargs_list=kwargs_list)

####################################################
####                  NETWORK                    ###
####################################################

"""
	here we specify the neural network.

"""

# net = Unet_bn(img_channels=data_channels, truth_channels=truth_channels, cost_dict_list=losses_dict, **kwargs)


####################################################
####                 TRAINING                    ###
####################################################
from scadec_Kraken.train_Kraken import Trainer_bn

# args for training
batch_size = kwargs.get("batch_size", 5) # batch size for training
valid_size = kwargs.get("valid_size", 5)  # batch size for validating

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
