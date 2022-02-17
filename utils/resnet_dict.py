import numpy as np

def get_dict(model_name):
    model_dict = dict()
    if model_name == 'resnet20':
        model_dict['output_size'] = [32*32]*8 + [16*16]*6 + [8*8]*6 + [1]
        model_dict['input_size'] = [32*32]*8 + [16*16]*6 + [8*8]*5 + [8*8+1,1] # 8*8+1 is from avgpool(x) and stage_3(x)
        model_dict['kernel_size'] = [3*3]*19 + [1]
        model_dict['s_list'] = [1,3,5,7,9,11,13,15,17]
        model_dict['t_list'] = [3,5,7,9,11,13,15,17,19]
        model_dict['ds_flops'] = 0
    
    elif model_name =='resnet32':
        model_dict['output_size'] = [32*32]*12 + [16*16]*10 + [8*8]*10 + [1]
        model_dict['input_size'] = [32*32]*12 + [16*16]*10 + [8*8]*9 + [8*8+1,1] # 8*8+1 is from avgpool(x) and stage_3(x)
        model_dict['kernel_size'] = [3*3]*31 + [1]
        model_dict['s_list'] = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]
        model_dict['t_list'] = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
        model_dict['ds_flops'] = 0
    
    elif model_name == 'resnet56':
        model_dict['output_size'] = [32*32]*20 + [16*16]*18 + [8*8]*18 +[1]
        model_dict['input_size'] = [32*32]*20 + [16*16]*18 + [8*8]*17 + [8*8+1,1] # 8*8+1 is from avgpool(x) and stage_3(x)
        model_dict['kernel_size'] = [3*3]*55 + [1]
        model_dict['s_list'] = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29
                ,31,33,35,37,39,41,43,45,47,49,51,53]
        model_dict['t_list'] = [3,5,7,9,11,13,15,17,19,21,23,25,27,29
                ,31,33,35,37,39,41,43,45,47,49,51,53,55]
        model_dict['ds_flops'] = 0
    
    elif model_name == 'resnet8':
        model_dict['output_size'] = [32*32]*4 + [16*16]*2 + [8*8]*2 + [1]
        model_dict['input_size'] = [32*32]*4 + [16*16]*2 + [8*8]*2 + [8*8+1,1] # 8*8+1 is from avgpool(x) and stage_3(x)
        model_dict['kernel_size'] = [3*3]*7 + [1]
        model_dict['s_list'] = [1,3,5]
        model_dict['t_list'] = [3,5,7]
        model_dict['ds_flops'] = 0
    
    L = len(model_dict['output_size'])
    val_dict = dict()
    val_dict['quadratic_val_list_flops_4D'] = list()
    val_dict['quadratic_val_list_flops_2D'] = list()
    val_dict['quadratic_val_list_mem_4D'] = list()
    val_dict['quadratic_val_list_mem_2D'] = list()

    val_dict['const_flops'] = model_dict['ds_flops']
    for i in range(L-1):    
        val_dict['quadratic_val_list_mem_4D'].append(1.0)
        val_dict['quadratic_val_list_mem_2D'].append(1.0*model_dict['kernel_size'][i])
        val_dict['quadratic_val_list_flops_4D'].append(model_dict['output_size'][i+1])
        val_dict['quadratic_val_list_flops_2D'].append(model_dict['output_size'][i+1]*model_dict['kernel_size'][i])
    
    return model_dict, val_dict
