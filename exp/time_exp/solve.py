import os, sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from exp.time_exp.time_layer import get_scale, load_info
from utils.read_op import read_pkl
from utils.write_op import write_pkl, create_muldir

import numpy as np
import argparse

def linear_solve(ST, y, debug=False):
    '''
    ST - np.array([nvar, ndata])
    y - np.array([ndata])
    '''
    S = np.transpose(ST, [1,0])
    b = np.reshape(y, [-1,1])

    A = np.matmul(ST, S)

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.inv(A+1e-5*np.eye(np.shape(A)[0]))

    x = np.matmul(A_inv, np.matmul(ST, b))
    bstar = np.matmul(S,x)
    mpe = np.mean(np.abs(np.divide(bstar-b, b)))
    r = 1-np.sum(np.square(b-bstar))/np.sum(np.square(b-np.mean(b)))
    if debug:
        print("mean error percent : {}".format(mpe))
        print("R^2 : {}".format(r))
    return x, mpe, r

def solve_layer(loadpath='./exp/time_exp/dict.pkl', loadpath_t='./time.pkl', savepath='./solution.pkl', model='M5', debug=False):
    info_list, input_shape_list, output_shape_list = load_info(path=loadpath)
    nlayer = len(info_list)

    content = read_pkl(loadpath_t)
    solution_dict = dict()
    mpe_list, r_list = [], []
    for idx, info in enumerate(info_list):
        key = info['key'] 
        if 'fc' in key: continue
        inf_h, inf_w = input_shape_list[idx]
        outf_h, outf_w = output_shape_list[idx]
        nin = info['inch']
        nout = info['outch']
        nker = info['kernel_size'][0]

        inscale = get_scale(inf_h)
        outscale = get_scale(outf_h)

        if nin < inscale:
            if nout < outscale:
                print("It is enough ({}/{}),({}/{})".format(nin, inscale, nout, outscale))
            else:
                x1 = list()
                x2 = list()
                y = list()
                for actoutidx in range(1, nout//outscale+1):
                    x1.append(1)
                    x2.append(actoutidx)
                    time = content['{}_out_{}'.format(idx, actoutidx)]
                    y.append(time)
                ST = np.array([x1,x2])
                y = np.array(y)
        else:
            if nout < outscale:
                x1 = list()
                x2 = list()
                y = list()
                for actinidx in range(1,nin//inscale+1):
                    time = content['{}_in_{}'.format(idx, actinidx)]
                    x1.append(1)
                    x2.append(actinidx)
                    y.append(time)
                ST = np.array([x1, x2])
                y=np.array(y)
            else:
                if model == 'M5':
                    # M5. linear regression with 1, nin, nin*nout.
                    x1 = list()
                    x2 = list()
                    x3 = list()
                    y = list()

                    for actinidx in range(1, nin//inscale+1):
                        for actoutidx in range(1, nout//outscale+1):
                            time = content['{}_{}_{}'.format(idx, actinidx, actoutidx)]
                            x1.append(1)
                            x2.append(actinidx)
                            x3.append(actinidx*actoutidx)
                            y.append(time)
                    ST = np.array([x1, x2, x3])
                
                elif model == 'M6':
                    # M6. linear regression with 1, nin, nout, nin*nout
                    x1 = list()
                    x2 = list()
                    x3 = list()
                    x4 = list()
                    y = list()

                    for actinidx in range(1, nin//inscale+1):
                        for actoutidx in range(1, nout//outscale+1):
                            time = content['{}_{}_{}'.format(idx, actinidx, actoutidx)]
                            x1.append(1)
                            x2.append(actinidx)
                            x3.append(actoutidx)
                            x4.append(actinidx*actoutidx)
                            y.append(time)
                    ST = np.array([x1, x2, x3, x4])
                else:
                    raise NotImplementedError

                y = np.array(y)
        solution, mpe, r = linear_solve(ST=ST, y=y, debug=debug)
        solution_dict[idx] = solution
        mpe_list.append(mpe)
        r_list.append(r)
    if debug:
        print("avg mpe : {}".format(sum(mpe_list)/len(mpe_list)))
    write_pkl(content=solution_dict, path=savepath)
    return mpe_list, r_list

def summary(mpe_list, r_list, loadpath='./exp/time_exp/dict.pkl'):
    info_list, input_shape_list, output_shape_list = load_info(path=loadpath)
    print("layer, H, W, nin, nout, k, s, inscale, outscale, MPE, R^2")
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

        inscale = get_scale(inf_h)
        outscale = get_scale(outf_h)
        print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {:.2f}, {:.2f}".format(idx+1,inf_h,inf_w, nin, nout, nker, stride, inscale, outscale, 100*mpe_list[idx], r_list[idx]))
    print("Average MPE : {:.2f} Average R^2 : {:.2f}".format(sum(mpe_list)/len(mpe_list)*100, sum(r_list)/len(r_list)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quad-model',default='M6',type=str,help='quadratic model type')
    parser.add_argument('--use-basic',action='store_true')
    args = parser.parse_args() # parameter required for model
    METADIR = './exp/time_exp/meta/'
    create_muldir(METADIR)

    if args.use_basic:
        loadpath_t = METADIR+'t_layer_basic.pkl'
        savepath = METADIR+'solution_basic_{}.pkl'.format(args.quad_model)
    else:
        loadpath_t = METADIR+'t_layer_general.pkl'
        savepath = METADIR+'solution_general_{}.pkl'.format(args.quad_model)
    print("pkl loaded from {} and the result will be saved in {}".format(loadpath_t, savepath))
    mpe_list, r_list = solve_layer(loadpath_t=loadpath_t, savepath=savepath, model=args.quad_model)
    summary(mpe_list, r_list)
