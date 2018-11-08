# Configurations of experiments
####################################################
####                DATA LOADING                 ###
####################################################

"""
	here loads all the data we need for training and validating.

"""

#parameter_str = '15-l2_masked-3C_motion-gradient_masked_mid5-reg_no-drop_0.9-FULL_SEG'
para_str_15 = 'Idx_15-Loss_l2_masked_mid5-Loss-gradient_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_15 = {
    'idx':15,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
}

para_str_15_2 = 'Idx_15_2-Loss_l2_masked_mid5-Loss-gradient_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Opti-Clipping'
para_dict_15_2 = {
    'idx':15,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam_clip'
}
# Still got NaN -> use loss cliping (upper_bound for gradient loss) in 

#l2_3C_motion_masked_mid5-Sobel_masked_mid5_w2-reg_no-drop_0.9-FULL_SEG
para_str_16 = 'Idx_16-Loss_l2_masked_mid5-Loss-Sobel_masked_mid5_w2-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_16 = {
    'idx':16,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'Sobel',
        'weight':50,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
}

# l2_3C_motion_masked_mid5-Sobel_masked_mid5-reg_no-drop_0.75-FULL_SEG
para_str_17 = 'Idx_17-Loss_l2_masked_mid5-Loss-Sobel_masked_mid5-Reg_no-Drop_0.75-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_17 = {
    'idx':16,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'Sobel',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.75,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    }
}


# Re-run 18 and 16 and 17
para_str_19 = 'Idx_19-Loss_l2_masked_mid5-Loss-Sobel_masked_mid5_w2-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_19 = para_dict_16
para_dict_19['GPU_IND'] = '3'

# From 15
para_str_20 = 'Idx_20-Loss_l2_masked_mid5-Loss-gradient_masked_mid5_Clip-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Opti_Clipping'
para_dict_20 = {
    'idx':20,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False,
        'upper_bound': 0.1
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam_clip',
    'GPU_IND':'2'
}

# 	Loss_l2_masked_mid5-Loss-LoG_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG
para_str_21 = 'Idx_21-Loss_l2_masked_mid5-Loss-LoG_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_21 = {
    'idx':21,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'LoG',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False,
        'upper_bound': 0.1
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam_clip',
    'GPU_IND':'3'
}

para_str_22 = 'Idx_22-Loss_l2_masked_mid5-Loss_LoG_masked_mid5_w2-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_22 = {
    'idx':22,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'LoG',
        'weight':50,
        'mask':'mid5',
        'mask_before_operate':False
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'3'
}

para_str_23 = 'Idx_23-Loss_l2_masked_mid5-Loss-gradient_XY_masked_mid5-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_23 = {
    'idx':23,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'mid5',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'2'
}

para_str_24 = 'Idx_24-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.9-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_24 = {
    'idx':24,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.9,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'2'
}

para_str_25 = 'Idx_22-Loss_l2_masked_norm-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_25 = {
    'idx':25,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'norm'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'3'
}

# Redo 24 with lower keep rate
para_str_26 = 'Idx_26-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_26 = {
    'idx':26,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'optimizer': 'adam',
    'GPU_IND':'2'
}

para_str_27 = 'Idx_27-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8-Proc_ero_2'
para_dict_27 = {
    'idx':27,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'proc_dict':{
        'data':{'erosion':{'size':(2,2)}},
        'truth':{}        
    },
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

para_str_28 = 'Idx_28-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8-Proc_ero_3'
para_dict_28 = {
    'idx':28,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'proc_dict':{
        'data':{'erosion':{'size':(3,3)}},
        'truth':{}        
    },
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

para_str_29 = 'Idx_29-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm-Reg_no-Drop_0.7-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_29 = {
    'idx':29,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        }],
    'reg':None,
    'Keep':0.7,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

para_str_30 = 'Idx_30-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Proc_ero_4'
para_dict_30 = {
    'idx':30,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True
    },
    'proc_dict':{
        'data':{'erosion':{'size':(4,4)}},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

para_str_31 = 'Idx_31-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_31 = {
    'idx':31,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        # "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'n_classes':8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },    
    'epochs':220,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}


# Re-run 29 without gradient in last 20 epochs
# para_str_32 = 'Idx_21-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end_20-Reg_no-Drop_0.7-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_str_32 = 'Idx_32-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end_20-Reg_no-Drop_0.7-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG'
para_dict_32 = {
    'idx':32,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_after':200
        }],
    'reg':None,
    'Keep':0.7,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        'structure':'Nagini'# Or Hydra
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,    
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# Re-run 31 with only l2 loss without mask
para_str_33 = 'Idx_33-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra_8'
para_dict_33 = {
    'idx':33,
    'losses':[
        {
        'name':'l2',
        'weight':1}],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        # "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'n_classes':8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },    
    'epochs':200,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'2'
}

para_str_34 = 'Idx_34-Loss_l2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra1_8'
para_dict_34 = {
    'idx':34,
    'losses':[
        {
        'name':'l2',
        'weight':1}],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        # "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        # 'structure': 'Nagini',
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

para_str_35 = 'Idx_35-Loss_l2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra3_8'
para_dict_35 = {
    'idx':35,
    'losses':[
        {
        'name':'l2',
        'weight':1}],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        # "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 3,
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# Re-run 32 with corrected net structure
para_str_36 = 'Idx_36-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra3_8'
para_dict_36 = {
    'idx':36,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':2,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_last':20
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        # "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "layers":3,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 3,
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# Re-run 34 with corrected net structure(layers = 5) and gradient loss
para_str_37 = 'Idx_37-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_10_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra1_8'
para_dict_37 = {
    'idx':37,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_after':200
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 1,
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# Should be 38!
para_str_38 = 'Idx_37-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_10_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra3_8'
para_dict_38 = {
    'idx':38,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_after':200
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 3,
        'n_classes': 8
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

para_str_39 = 'Idx_39-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_10_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_39 = {
    'idx':39,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_after':200
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':220,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# re-run 32(i.e. "21") with only 5 additional epochs
para_str_1018 = 'Idx_1018-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_w_10_invalid_end5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Nagini'
para_dict_1018 = {
    'idx':1018,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        # 'invalid_after':200
        'invalid_last': 5
        }],
    'reg':None,
    'Keep':0.7,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        'structure':'Nagini'# Or Hydra
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':205,    
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# Re-run 39 without gradient mask and last 20
para_str_40 = 'Idx_40-Loss_l2_masked_mid5-Loss-gradient_XY_w_10-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_40 = {
    'idx':40,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# rerun 39 with NMS
para_str_41 = 'Idx_41-Loss_l2_masked_mid5-Loss-gradient_XY_masked_norm_NMS_w_10_invalid_end20-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_41 = {
    'idx':41,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':10,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
        'NMS': True,
        'NMS_window_size': 3,
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# re-run 40 with different gradient weight
para_str_42 = 'Idx_42-Loss_l2_masked_mid5-Loss-gradient_XY_w_30-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_42 = {
    'idx':42,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':30,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# Add gradient weight
para_str_43 = 'Idx_43-Loss_l2_masked_mid5-Loss-gradient_XY_w_50-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_43 = {
    'idx':43,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':50,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# new gradient loss
para_str_44 = 'Idx_44-Loss_l2_masked_mid5-Loss_gradient_type_2_w_30-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_44 = {
    'idx':44,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':30,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
	'type':'2',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# Another new gradient loss
para_str_45 = 'Idx_45-Loss_l2_masked_mid5-Loss_gradient_type_3_w_30-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_45 = {
    'idx':45,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':30,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
	'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# Rerun 45 with different gradient loss
para_str_46 = 'Idx_46-Loss_l2_masked_mid5-Loss_gradient_type_3_w_100-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_46 = {
    'idx':46,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':100,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'no_GAN_net_func':True,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# Add perceptual loss (patchGAN)
para_str_47 = 'Idx_47-Loss_l2_masked_mid5-Loss_gradient_type_3_w_30-Loss_patchPerceptual_w_01-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_47 = {
    'idx':47,
    'losses':[
        {
            'name':'l2',
            'weight':1,
            'mask':'mid5'},
        {
            'name':'edge',
            'edge_type':'gradient',
            'weight':30,
            'mask':None,
            'mask_before_operate':False,
            'get_XY':True,
            'type':'3',
            'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'GAN': True,
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam_patchPerceptual',
    'server': '1',
    'GPU_IND':'3'
}

# Re-run only 3C + mid5 mask
# l2_3C_motion_masked_reg_no_drop_0.75_FULL_data
para_str_48 = 'Idx_48-Loss_l2_masked_mid5-Reg_no-Drop_0.75-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Nagini'
para_dict_48 = {
    'idx':48,
    'losses':[
        {
            'name':'l2',
            'weight':1,
            'mask':'mid5'}
        ],
    'reg':None,
    'Keep':0.75,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 5,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        #'structure': 'Nagini'
        'structure':{
                    'type':'Nagini',
        }
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# Test GAN
para_str_49 = 'Idx_49-GAN_test'
para_dict_49 = {
    'idx':49,
    'losses':[
        {
            'name':'l2',
            'weight':1,
            'mask':'mid5'},
        {
            'name':'edge',
            'edge_type':'gradient',
            'weight':30,
            'mask':None,
            'mask_before_operate':False,
            'get_XY':True,
            'type':'3',
            'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 2,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 32,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Nagini',
            'GAN':True,
            'Ouroboros':True,
            'neck_len': 2,
            'n_classes': 16
        }
        # 'structure': 'Nagini',
        # 'GAN': True,
        # 'neck_len': 2,
        # 'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam_patchPerceptual',
    'server': '1',
    'GPU_IND':'3'
}

# re-run 46 with gradient mask
para_str_50 = 'Idx_50-Loss_l2_masked_mid5-Loss_gradient_type_3_w_50-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_50 = {
    'idx':50,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':50,
        'mask':'norm',
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure': 'Hydra',
        'neck_len': 2,
        'n_classes': 16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# re-run 46 with Ouroboros structure
para_str_51 = 'Idx_51-Loss_l2_masked_mid5-Loss_gradient_type_3_w_100-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16_Ouroboros'
para_dict_51 = {
    'idx':51,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':100,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':True,
            'neck_len': 2,
            'n_classes': 16
        },
        # 'no_GAN_net_func':True,
        # 'structure':'Hydra',
        # 'neck_len': 2,
        # 'n_classes':16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# re-run ?(masked l2 and gradient) with smaller network
para_str_52 = 'Idx_52-Loss_l2_masked_mid5-Loss_gradient_type_3_w_50-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Nagini_Ouroboros'
para_dict_52 = {
    'idx':52,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':50,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 32,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 5,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Nagini',
            'GAN':False,
            'Ouroboros':True,
        },
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}
# result: strangely grey. 

# K-space l2 loss
para_str_53 = 'Idx_53-Loss_l2_masked_mid5-Loss_kl2_w10_Loss_gradient_type_3_w_50-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_53 = {
    'idx':53,
    'losses':[
        {
            'name':'l2',
            'weight':1,
            'mask':'mid5'
        },
        {
            'name':'edge',
            'edge_type':'gradient',
            'weight':50,
            'mask':None,
            'mask_before_operate':False,
            'get_XY':True,
            'type':'3',
            'invalid_last':0
        },{
            'name':'kl2',
            'weight':10
        }
        ],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 32,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 5,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':False,
            'neck_len': 2,
            'n_classes': 16
        },
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# re-run 53 with high pass in k-space
para_str_54 = 'Idx_54-Loss_l2_masked_mid5-Loss_kl2_w_10_mask_GaussianHighPass_Loss_gradient_type_3_w_50-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_54 = {
    'idx':54,
    'losses':[
        {
            'name':'l2',
            'weight':1,
            'mask':'mid5'
        },
        {
            'name':'edge',
            'edge_type':'gradient',
            'weight':50,
            'mask':None,
            'mask_before_operate':False,
            'get_XY':True,
            'type':'3',
            'invalid_last':0
        },{
            'name':'kl2',
            'weight':10,
            'mask':'GaussianHighPass'
        }
        ],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 32,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 5,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':False,
            'neck_len': 2,
            'n_classes': 16
        },
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# re-run 51 with different structure
para_str_55 = 'Idx_55-Loss_l2_masked_mid5-Loss_gradient_type_3_w_100-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra3_3_16_Ouroboros'
para_dict_55 = {
    'idx':55,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':100,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 3,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':True,
            'neck_len': 3,
            'n_classes': 16
        },
        # 'no_GAN_net_func':True,
        # 'structure':'Hydra',
        # 'neck_len': 2,
        # 'n_classes':16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# re-run 51 with another different structure
para_str_56 = 'Idx_56-Loss_l2_masked_mid5-Loss_gradient_type_3_w_100-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra2_4_16_Ouroboros'
para_dict_56 = {
    'idx':56,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':'mid5'},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':100,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 2,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':True,
            'neck_len': 4,
            'n_classes': 16
        },
        # 'no_GAN_net_func':True,
        # 'structure':'Hydra',
        # 'neck_len': 2,
        # 'n_classes':16
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}

# re-run 54 with l1 k-space loss
para_str_57 = 'Idx_57-Loss_l2_masked_mid5-Loss_kl2_w_10_mask_GaussianHighPass_Loss_gradient_type_3_w_50-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Hydra4_2_16'
para_dict_57 = {
    'idx':57,
    'losses':[
        {
            'name':'l2',
            'weight':1,
            'mask':'mid5'
        },
        {
            'name':'edge',
            'edge_type':'gradient',
            'weight':50,
            'mask':None,
            'mask_before_operate':False,
            'get_XY':True,
            'type':'3',
            'invalid_last':0
        },{
            'name':'kl1',
            'weight':10,
            'mask':'GaussianHighPass'
        }
        ],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 32,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 5,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':False,
            'neck_len': 2,
            'n_classes': 16
        },
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# OctoNet
para_str_58 = 'Idx_58-Loss_l2_masked_mid5-Loss_gradient_type_3_w_50-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-Nagini_Octo'
para_dict_58 = {
    'idx':58,
    'losses':
    [
        [
            'forHighPass',        
            {
                'name':'l2',                
                'weight':1,
                'mask':'mid5',
                'mask_before_operate':False,
                'get_XY':True,
                'type':'3',
                'invalid_last':0
            }
        ],
        [
            'forLowPass',        
            {
                'name':'l2',
                'weight':1,
                'mask':'mid5'
            }
        ],[
            'forRecon',        
            {
                'name':'l2',
                'weight':1,
                'mask':'mid5'
            }
        ]

    ],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'kwargs' : [{             # Kwargs[0] must be common parameters
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        "structure_type":"highLowPass"
    },
    {
        "name":"lowPass",
        "layers": 3,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 32,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 5,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling        
        'structure':{
            'type':'Nagini',
            'GAN':False,
            'Ouroboros':False,
            'n_classes':16
        },
    },
    {
        "name":"highPass",
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 32,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 5,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        'structure':{
            'type':'Nagini',
            'GAN':False,
            'Ouroboros':False,
            'n_classes':16
        },
    }],
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '2',
    'GPU_IND':'3'
}

# re-run 51 without some classes
para_str_59 = 'Idx_59-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'
para_dict_59 = {
    'idx':59,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':None},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':100,
        'mask':'mid5',
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'ignore_classes':[0,1,13,14,15],
    # 'ignore_classes':[14,15],
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':True,
            'neck_len': 2,
            'n_classes': 16
        },
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# re-run 59 with higher center mask value
para_str_60 = 'Idx_60-Loss_l2-Loss_gradient_type_3_w_100_masked_mid7-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'
para_dict_60 = {
    'idx':60,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':None},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':100,
        'mask':'mid7',
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'ignore_classes':[0,1,13,14,15],
    # 'ignore_classes':[14,15],
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':True,
            'neck_len': 2,
            'n_classes': 16
        },
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# re-run 59 with precomputed masks
para_str_61 = 'Idx_61-Loss_l2-Loss_gradient_type_3_w_100-Loss_preMask-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'
para_dict_61 = {
    'idx':61,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':None},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':50,
        'mask':None,
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'preMask':True,
    'ignore_classes':[0,1,13,14,15],
    # 'ignore_classes':[14,15],
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':True,
            'neck_len': 2,
            'n_classes': 16
        },
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}

# re-run 61 with mask
para_str_62 = 'Idx_62-Loss_l2-Loss_gradient_type_3_w_100_mask_mid5-Loss_preMask-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'
para_dict_62 = {
    'idx':62,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':None},
        {
        'name':'edge',
        'edge_type':'gradient',
        'weight':100,
        'mask':'mid5',
        'mask_before_operate':False,
        'get_XY':True,
	    'type':'3',
        'invalid_last':0
        }],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'preMask':True,
    'ignore_classes':[0,1,13,14,15],
    # 'ignore_classes':[14,15],
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':True,
            'neck_len': 2,
            'n_classes': 16
        },
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'2'
}
# rerun 61 with different +1
para_str_63 = 'Idx_63-Loss_l2-Loss_preMask-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'
para_dict_63 = {
    'idx':63,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':None},
        ],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'Gt':'FULL_SEG',
    'preMask':True,
    'preMaskAdd':0.1,
    'ignore_classes':[0,1,13,14,15],
    # 'ignore_classes':[14,15],
    'kwargs' : {
        "layers": 4,           # how many resolution levels we want to have
        "conv_times": 2,       # how many times we want to convolve in each level
        "features_root": 64,   # how many feature_maps we want to have as root (the following levels will calculate the feature_map by multiply by 2, exp, 64, 128, 256)
        "filter_size": 3,      # filter size used in convolution
        "pool_size": 2,        # pooling size used in max-pooling
        "summaries": True,
        "get_loss_dict": True,
        "batch_size": 5,
        "valid_size": 5,
        'structure':{
            'type':'Hydra',
            'GAN':False,
            'Ouroboros':True,
            'neck_len': 2,
            'n_classes': 16
        },
    },
    'proc_dict':{
        'data':{},
        'truth':{}        
    },
    'epochs':200,
    'optimizer': 'adam',
    'server': '1',
    'GPU_IND':'3'
}


#para_dict_use = para_dict_42
#para_str_use = para_str_42
# para_dict_use_train = para_dict_58
# para_str_use_train = para_str_58
para_dict_use_train = para_dict_63
para_str_use_train = para_str_63

para_dict_use_test = para_dict_46
para_str_use_test = para_str_46
