import os, sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

torch.backends.cudnn.enabled = False

import torch
import argparse

from utils.read_op import read_pkl 
from utils.write_op import write_pkl, create_muldir 
from utils.time_op import MeasureTime
from networks.conv_model import GeneralConv, BasicConv

def get_scale(nf):
    if nf>=28:
        return 16
    elif nf>=14:
        return 32
    elif nf>=7:
        return 64

def load_info(path='dict.pkl'):
    content = read_pkl(path)
    info_list = content['info_list']
    input_shape_list = content['input_shape_list']
    output_shape_list = content['output_shape_list']
    return info_list, input_shape_list, output_shape_list

def get_time(model, input_shape=[1,3,224,224], name="", rep=1000):
    dummy_input = torch.randn(*input_shape).cuda()

    nimble_model = torch.cuda.Nimble(model)
    nimble_model.prepare(dummy_input, training=False)

    # Execute the object
    dummy_input = [torch.rand(*input_shape).cuda() for _ in range(1000)]
    rand_input = [torch.rand(*input_shape).cuda() for _ in range(rep)]
    with torch.no_grad():
        for i in range(1000):
            output = nimble_model(dummy_input[i])
        with MeasureTime(name) as t:
            for i in range(rep):
                output = nimble_model(rand_input[i])
            cur_time = t.get_time()
            elapsed_time = cur_time/rep*1000
            del output
            print("{} : {} ms".format(name, elapsed_time))
    del rand_input, nimble_model, model, dummy_input
    return  elapsed_time

def measure_time_conv(nin, nout, nker, pad, stride, bias, inf, key, groups=1):
    input_shape = [1, nin, inf, inf]
    model = GeneralConv(nin=nin, nout=nout, nker=nker, pad=pad, stride=stride, bias=bias)
    model = model.cuda().eval()
    time = get_time(model=model, input_shape=input_shape, name=key)
    return time 

def measure_time_conv_basic(nin, nout, nker, pad, stride, bias, inf, key, groups=1):
    input_shape = [1, nin, inf, inf]
    model = BasicConv(nin=nin, nout=nout, nker=nker, pad=pad, stride=stride, bias=bias)
    model = model.cuda().eval()
    time = get_time(model=model, input_shape=input_shape, name=key)
    return time 

def layer_exp(loadpath='./dict.pkl', savepath='./time.pkl', measure_func=measure_time_conv, resume=False):
    info_list, input_shape_list, output_shape_list = load_info(path=loadpath)
    if resume:
        try:
            # Load content.
            content = read_pkl(savepath)
            print("loaded idx : {} / {}".format(content['idx'], len(info_list)))
        except:
            print("no file in path {}".format(savepath))
    else:
        # Init content.
        content = dict()
        content['idx'] = -1
    
    # Measure wall-clock time of each conv layer in ResNet-50 for various nin, nout.
    for idx, info in enumerate(info_list):
        # Resume.
        if resume and idx <= content['idx']: continue
        
        # Get info.
        key = info['key'] 
        if 'fc' in key: continue
        inf_h, inf_w = input_shape_list[idx]
        outf_h, outf_w = output_shape_list[idx]
        nin, nout, nker, stride, pad, bias = info['inch'], info['outch'], info['kernel_size'][0], info['stride'][0], info['padding'][0], info['bias']
        inscale, outscale = get_scale(inf_h), get_scale(outf_h)

        if nin < inscale:
            if nout < outscale:
                print("It is enough ({}/{}),({}/{})".format(nin, inscale, nout, outscale))
            else:
                for actoutidx in range(1, nout//outscale+1):
                    nact_in = nin
                    nact_out = outscale * actoutidx
                    time = measure_time_conv(nin=nact_in, nout=nact_out, nker=nker, pad=pad, stride=stride, bias=bias, inf=inf_h, key=key)
                    content['{}_out_{}'.format(idx, actoutidx)] = time
        else:
            if nout < outscale:
                for actinidx in range(1, nin//inscale+1):
                    nact_in = inscale*actinidx
                    nact_out = nout 
                    time = measure_time_conv(nin=nact_in, nout=nact_out, nker=nker, pad=pad, stride=stride, bias=bias, inf=inf_h, key=key)
                    content['{}_in_{}'.format(idx, actinidx)] = time
            else:
                for actinidx in range(1, nin//inscale+1):
                    for actoutidx in range(1, nout//outscale+1):
                        nact_in = inscale * actinidx
                        nact_out = outscale * actoutidx
                        time = measure_time_conv(nin=nact_in, nout=nact_out, nker=nker, pad=pad, stride=stride, bias=bias, inf=inf_h, key=key)
                        content['{}_{}_{}'.format(idx, actinidx, actoutidx)] = time
        # For resume
        content['idx'] = idx
        write_pkl(content=content, path=savepath) 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-basic',action='store_true')
    args = parser.parse_args() # parameter required for model
    
    METADIR = './exp/time_exp/meta/'
    create_muldir(METADIR)

    if args.use_basic:
        layer_exp(loadpath='./exp/time_exp/dict.pkl', savepath=METADIR+'t_layer_basic.pkl', measure_func=measure_time_conv_basic)
    else:
        layer_exp(loadpath='./exp/time_exp/dict.pkl', savepath=METADIR+'t_layer_general.pkl', measure_func=measure_time_conv)
    
    
