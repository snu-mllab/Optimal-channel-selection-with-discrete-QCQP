import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import torch
import numpy as np

from utils.exp_2D_cplex_op import single_quadratic_constraint_skip_solver
from utils.exp_4D_cplex_op import fast_quadratic_4D_skip_solver
from utils.greedy_solver import SkipFastCplexSolver, SkipFastCplex4DSolver
from utils.write_op import write_pkl
from utils.read_op import read_pkl

def prun_skip_2D_single(
        array_list, 
        s_list, 
        t_list, 
        quadratic_val_list, 
        max_constraint,
        use_cplex=False,
        timelimit=None,
        path=None):
    '''
    Args:
        array_list - list of Numpy 2D [nout, nin] size : nlist
        s_list - list of int size : N
        t_list - list of int size : N
        quadratic_val_list - list of int size : nlist
        max_constraint - float
        use_cplex - bool
        timelimit - float
        iterlimit - int
    Return:
        binary_list - list of Numpy 2d [nout, nin] size : nlist
    '''
    if path is not None: print("path:{}".format(path))

    L = len(array_list)
    try:
        u_dict=read_pkl(path+'u_dict.pkl')
        v_dict=read_pkl(path+'v_dict.pkl')
    except OSError:
        if use_cplex:
            u_dict, v_dict = single_quadratic_constraint_skip_solver(
                    array_list=array_list,
                    quadratic_val_list=quadratic_val_list,
                    skip_s_list=s_list,
                    skip_t_list=t_list,
                    max_constraint=max_constraint,
                    olamb=10.0,
                    clamb=10.0,
                    keep_idx=[0],
                    timelimit=timelimit,
                    debug=True)
        else:
            min_select_list = [3] + [4]*L
            sfcs = SkipFastCplexSolver(
                    array_list=array_list,
                    min_select_list=min_select_list,
                    skip_s_list=s_list,
                    skip_t_list=t_list, 
                    is_imagenet=False
                    )
            sfcs.set_val_list(quadratic_val_list)
            sfcs.init_custom(divide=3)
            sfcs.greedy_fill(max_cons=max_constraint)
            u_dict, v_dict, binary_array_list = sfcs.get_solution()
        if path is not None:
            write_pkl(u_dict, path+'u_dict.pkl')
            write_pkl(v_dict, path+'v_dict.pkl')
        else: 
            print(f"{path} is None")

    cons  = 0
    for lidx, array in enumerate(array_list):
        cons += np.sum(u_dict[lidx+1]) * np.sum(v_dict[lidx]) * quadratic_val_list[lidx]
    print("constraint : {}".format(cons))
    return u_dict, v_dict

def prun_skip_4D(
        array_list, 
        s_list, 
        t_list, 
        quadratic_val_list, 
        max_constraint,
        gamma,
        timelimit=None,
        use_cplex=False,
        path=None
        ):
    '''
    Args:
        array_list - list of Numpy 2D [nout, nin] size : nlist
        s_list - list of int size : N
        t_list - list of int size : N
        quadratic_val_list - list of int size : nlist
        max_constraint - float
        gamma - float
        timelimit - float
        path - string
    Return:
        binary_list - list of Numpy 2d [nout, nin] size : nlist
    '''
    L = len(array_list)
    try:
        u_dict=read_pkl(path+'u_dict.pkl')
        v_dict=read_pkl(path+'v_dict.pkl')
        q_dict=read_pkl(path+'q_dict.pkl')
    except OSError:
        L = len(array_list)

        if use_cplex:
            solver = single_quadratic_constraint_skip_solver
            u_dict, v_dict, q_dict, binary_array_list = fast_quadratic_4D_skip_solver(
                    array_list=array_list,
                    quadratic_val_list=quadratic_val_list,
                    skip_s_list=s_list,
                    skip_t_list=t_list,
                    max_constraint=max_constraint, 
                    solver=solver,
                    gamma=gamma,
                    timelimit=timelimit,
                    debug=True)
        else:
            min_select_list = [3] + [4]*L
            sfcs = SkipFastCplex4DSolver(
                    array_list=array_list,
                    min_select_list=min_select_list,
                    skip_s_list=s_list,
                    skip_t_list=t_list, 
                    is_imagenet=False
                    )
            sfcs.set_val_list(quadratic_val_list)
            sfcs.init_custom(divide=3)
            sfcs.greedy_fill(max_cons=max_constraint)
            u_dict, v_dict, q_dict, binary_array_list = self.get_solution()

        if path is not None:
            write_pkl(u_dict, path+'u_dict.pkl')
            write_pkl(v_dict, path+'v_dict.pkl')
            write_pkl(q_dict, path+'q_dict.pkl')
        else: 
            print("path is None") 
    return u_dict, v_dict, q_dict

