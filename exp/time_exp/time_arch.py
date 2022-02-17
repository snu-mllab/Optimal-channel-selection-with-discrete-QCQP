import os, sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

torch.backends.cudnn.enabled = False

import torch
import torch.optim as optim
import numpy as np
import argparse

from utils.read_op import read_pkl 
from utils.write_op import write_pkl, create_muldir
from utils.time_op import MeasureTime
from networks.generate_resnet import gen_resnet50
from exp.time_exp.sample import gen_n_list 
from exp.time_exp.time_layer import get_time, get_scale

def load_info(path='./exp/time_exp/dict.pkl'):
    content = read_pkl(path)
    info_list = content['info_list']
    input_shape_list = content['input_shape_list']
    output_shape_list = content['output_shape_list']
    val_list = content['val_list']
    s_list = content['s_list']
    t_list = content['t_list']
    return info_list, input_shape_list, output_shape_list, val_list, s_list, t_list

def get_real_time50(n, name):
    m = gen_resnet50(ch_info=n[1:])
    m.eval().cuda()
    t = get_time(m,[1,3,224,224],name)
    return t

def estimate_infer(n, solution_dict, info_list, input_shape_list, output_shape_list, val_list, model):
    est_infer = 0

    for idx, info in enumerate(info_list):
        key = info['key'] 

        if 'fc' in key: continue
        inf_h, inf_w = input_shape_list[idx]
        outf_h, outf_w = output_shape_list[idx]
        nin = info['inch']
        nout = info['outch']
        nker = info['kernel_size'][0]
        stride = info['stride'][0]
        pad = info['padding'][0]
        bias = info['bias']
        val = val_list[idx]
        solution = np.reshape(solution_dict[idx], [-1])
        solution = [float(v) for v in solution]

        inscale = get_scale(inf_h)
        outscale = get_scale(outf_h)
        if nin < inscale:
            if nout < outscale:
                continue
            else:
                est_infer += solution[0] + solution[1] * n[idx+1]/outscale
        else:
            if nout < outscale:
                est_infer += solution[0] + solution[1] * n[idx]/inscale
            else:
                if model == 'M5':
                    est_infer += solution[0] + solution[1]*n[idx]/inscale + solution[2]*n[idx]*n[idx+1]/inscale/outscale
                elif model == 'M6':
                    est_infer += solution[0] + solution[1]*n[idx]/inscale + solution[2]*n[idx+1]/outscale + solution[3]*n[idx]*n[idx+1]/inscale/outscale
                else:
                    raise NotImplementedError

    return est_infer

def main(loadpath, loadpath_s, savepath, num_n=100, group=[[4,7,10],[13,16,19,22],[25,28,31,34,37,40],[43,46,49]], model='M5', resume=False):
    # Get infos
    info_list, input_shape_list, output_shape_list, val_list, s_list, t_list = load_info(path=loadpath)

    actinfo = [info['inch'] for info in info_list] + [1000]
    nlayer = len(info_list)
    print("actinfo : {}, nlayer : {}".format(actinfo, nlayer))
    
    solution_dict = read_pkl(loadpath_s)
    
    if resume:
        try:
            content = read_pkl(savepath)
            n_list = content['n_list']
            print("loaded idx : {} / {}".format(content['idx'], len(n_list)))
        except:
            print("no file in path {}".format(savepath))
    else:
        # Sample sparsity combinations.
        n_list = gen_n_list(actinfo, input_shape_list, num_n=num_n, group=group, n=nlayer+1)
        content = dict()
        content['idx'] = -1
        content['n_list'] = n_list
        content['est_infer'] = []
        content['wall_clock'] = []
        content['used_n'] = []
    
    for idx, n in enumerate(n_list):
        if resume and content['idx'] >= idx:
            continue
        print(idx, n)
        # Get actual wall-clock time and estimated inference time.
        wall_clock = get_real_time50(n, name='n_{}'.format(idx))
        est_infer = estimate_infer(n, solution_dict, info_list, input_shape_list, output_shape_list, val_list, model)

        print(wall_clock, est_infer)
        content['est_infer'].append(est_infer)
        content['wall_clock'].append(wall_clock)
        content['used_n'].append(n)
        content['idx'] = idx
        write_pkl(content, savepath)

def get_mpe(loadpath):
    content = read_pkl(loadpath)
    est_infer_list = content['est_infer']
    wall_clock_list = content['wall_clock']
    
    mpe = 0
    for est_infer, wall_clock in zip(est_infer_list, wall_clock_list):
        mpe += abs(est_infer - wall_clock)/wall_clock
    mpe /= len(est_infer_list)
    print("mpe = {}".format(mpe))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quad-model',default='M6',type=str,help='quadratic model type')
    parser.add_argument('--num-n',default=100,type=int,help='the number of samples')
    parser.add_argument('--use-basic',action='store_true')
    args = parser.parse_args() # parameter required for model
    METADIR = './exp/time_exp/meta/'
    create_muldir(METADIR)

    if args.use_basic:
        loadpath_s = METADIR + 'solution_basic_{}.pkl'.format(args.quad_model)
        savepath = METADIR + 't_arch_basic.pkl'
    else:
        loadpath_s = METADIR + 'solution_general_{}.pkl'.format(args.quad_model)
        savepath = METADIR + 't_arch_general.pkl'

    main(loadpath='./exp/time_exp/dict.pkl', loadpath_s=loadpath_s, savepath=savepath,num_n=args.num_n,model=args.quad_model)
    get_mpe(savepath)
