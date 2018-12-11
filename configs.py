para_str_72 = 'Idx_72-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T400-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'
para_dict_72 = {
    'idx':72,
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
    'ObSampleLineNum': 400,
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
    'server': '2',
    'GPU_IND':'2'
}

para_dict_73 = para_dict_72.copy()
para_dict_73['ObSampleLineNum'] = 800
para_dict_73['server'] = 2
para_dict_73['GPU_IND'] = '3'
para_str_73 = 'Idx_73-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T800-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'

para_dict_74 = para_dict_72.copy()
para_dict_74['ObSampleLineNum'] = 1200
para_dict_74['server'] = 1
para_dict_74['GPU_IND'] = '3'
para_str_74 = 'Idx_74-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T1200-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'

para_dict_75 = para_dict_72.copy()
para_dict_75['ObSampleLineNum'] = 1600
para_dict_75['server'] = 1
para_dict_75['GPU_IND'] = '2'
para_str_75 = 'Idx_75-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T1600-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'

# para_dict_76 = para_dict_72.copy()
# para_dict_76['ObSampleLineNum'] = 2000
# para_dict_76['server'] = 2
# para_dict_76['GPU_IND'] = '2'
# para_str_76 = 'Idx_76-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T2000-Gt_FULL_SEG-LessClass-Hydra4_2_16_Ouroboros'

para_str_77 = 'Idx_77-Loss_l2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T400-Gt_FULL_SEG-LessClass-Nagini_4_Ouroboros'
para_dict_77 = {
    'idx':77,
    'losses':[
        {
        'name':'l2',
        'weight':1,
        'mask':None}],
    'reg':None,
    'Keep':0.8,
    'Ob':'FULL_SEG_3C_motion',
    'ObSampleLineNum': 400,
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
            'type':'Nagini',
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

para_dict_78 = para_dict_77.copy()
para_dict_78['ObSampleLineNum'] = 800
para_dict_78['server'] = 2
para_dict_78['GPU_IND'] = '2'
para_str_78 = 'Idx_78-Loss_l2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T800-Gt_FULL_SEG-LessClass-Nagini_4_Ouroboros'

para_dict_79 = para_dict_77.copy()
para_dict_79['ObSampleLineNum'] = 1200
para_dict_79['server'] = 2
para_dict_79['GPU_IND'] = '3'
para_str_79 = 'Idx_79-Loss_l2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T1200-Gt_FULL_SEG-LessClass-Nagini_4_Ouroboros'

para_dict_80 = para_dict_77.copy()
para_dict_80['ObSampleLineNum'] = 1600
para_dict_80['server'] = '2'
para_dict_80['GPU_IND'] = '1'
para_str_80 = 'Idx_80-Loss_l2-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T1600-Gt_FULL_SEG-LessClass-Nagini_4_Ouroboros'

para_str_81 = 'Idx_81-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T400-Gt_FULL_SEG-LessClass-Nagini_4_Ouroboros'
para_dict_81 = {
    'idx':81,
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
    'ObSampleLineNum': 400,
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
            'type':'Nagini',
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
    'server': '2',
    'GPU_IND':'3'
}

para_dict_82 = para_dict_81.copy()
para_dict_82['ObSampleLineNum'] = 800
para_dict_82['server'] = '2'
para_dict_82['GPU_IND'] = '2'
para_str_82 = 'Idx_82-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T800-Gt_FULL_SEG-LessClass-Nagini_4_Ouroboros'

para_dict_83 = para_dict_81.copy()
para_dict_83['ObSampleLineNum'] = 1200
para_dict_83['server'] = '1'
para_dict_83['GPU_IND'] = '3'
para_str_83 = 'Idx_83-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T1200-Gt_FULL_SEG-LessClass-Nagini_4_Ouroboros'

para_dict_84 = para_dict_81.copy()
para_dict_84['ObSampleLineNum'] = 1600
para_dict_84['server'] = '1'
para_dict_84['GPU_IND'] = '2'
para_str_84 = 'Idx_84-Loss_l2-Loss_gradient_type_3_w_100_masked_mid5-Reg_no-Drop_0.8-Ob_FULL_SEG_3C_motion_T1600-Gt_FULL_SEG-LessClass-Nagini_4_Ouroboros'


para_dict_use_train = para_dict_80
para_str_use_train = para_str_80
para_dict_use_test = para_dict_72
para_str_use_test = para_str_72