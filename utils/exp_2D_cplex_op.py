import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.qcqp_support_op import get_nact_list
from utils.cplex_support import VarManager, LinearManager, LinearConstraintsManager, QuadraticManager, CplexSolver, seq_linear_const

import numpy as np

def single_quadratic_constraint_skip_solver(
        array_list,
        quadratic_val_list,
        skip_s_list,
        skip_t_list,
        max_constraint,
        olamb=None,
        clamb=10,
        keep_idx = [0],
        timelimit=None,
        is_imagenet=False,
        emph_memory=True,
        file_set=False,
        variable_select=False,
        node_select=True,
        debug=False):
    '''
    Without block
    With skip connection
        L - nlayer 
            array_list[lidx]

        L+1 - 
        nact_list
        e, f
    Args:
        array_list - list of 2D array [L]
        quadratic_val_list - list of int, [L]
        skip_s_list - list of int <=L
        skip_t_list - list of int <=L
        max_constraint - int
        olamb - float lambda for objective 
        clamb - float lambda for constraint
    '''
    L = len(array_list)

    nact_list = get_nact_list(array_list)
    if debug:
        print("nact_list : {}".format(nact_list))
        for sidx, tidx in zip(skip_s_list, skip_t_list):
            print("nact_s : {}, nact_t : {}".format(nact_list[sidx], nact_list[tidx]))

    cplexsolve = CplexSolver(timelimit=timelimit, emph_memory=emph_memory, file_set=file_set, variable_select=variable_select, node_select=node_select)

    vm = VarManager()
    var_idx = -1
    u2var_dict, v2var_dict = dict(), dict()
    # u^(l)_j, v^(l)_j
    for lidx, nact in enumerate(nact_list): 
        u2var_dict[lidx] = np.zeros([nact], dtype=np.int32)
        v2var_dict[lidx] = np.zeros([nact], dtype=np.int32)

        for actidx in range(nact):
            if lidx in keep_idx: 
                vm.add(name="u{}_{}".format(lidx, actidx), lower=1.0, upper=1.0)
            else:
                vm.add(name="u{}_{}".format(lidx, actidx), lower=0.0, upper=1.0)
            var_idx+=1
            u2var_dict[lidx][actidx] = var_idx

        if lidx not in skip_t_list:
            for actidx in range(nact):
                v2var_dict[lidx][actidx] = u2var_dict[lidx][actidx]
        else:
            for actidx in range(nact):
                if lidx in keep_idx:
                    vm.add(name="v{}_{}".format(lidx, actidx), lower=1.0, upper=1.0)
                else:
                    vm.add(name="v{}_{}".format(lidx, actidx), lower=0.0, upper=1.0)
                var_idx+=1
                v2var_dict[lidx][actidx] = var_idx

    cplexsolve.set_variable(vm)
    # ================================================================================ #
    # Linear constraint
    # ================================================================================ #
    linear_consman = LinearConstraintsManager()
    
    seq_linear_const(linear_consman, nact_list, u2var_dict, v2var_dict, skip_s_list, skip_t_list, is_imagenet)
    
    cplexsolve.set_linear_constraints(linear_consman)
    # ==================================================================#
    # Quadratic constraint
    # ==================================================================#
    qm = QuadraticManager()

    for lidx, array in enumerate(array_list):
        nout, nin = array.shape
        for outidx in range(nout): 
            for inidx in range(nin):
                # u^{(l)}_j 
                uvaridx = int(u2var_dict[lidx+1][outidx])
                # v^{(l-1)}_i
                vvaridx = int(v2var_dict[lidx][inidx])
                val = float(quadratic_val_list[lidx])
                qm.add_quad(q_ind1=uvaridx, q_ind2=vvaridx, q_val=val)
    if debug:
        qm.print_info()

    cplexsolve.set_quadratic_constraint(qm, rhs=max_constraint, sense="L", name="quadratic_constraint", clamb=clamb)
    # ================================================================================
    # Set objective
    # ================================================================================
    qm = QuadraticManager()

    for lidx, array in enumerate(array_list):
        nout, nin = array.shape
        for inidx in range(nin):
            for outidx in range(nout): 
                # u^{(l)}_j 
                uvaridx = int(u2var_dict[lidx+1][outidx])
                # v^{(l-1)}_i
                vvaridx = int(v2var_dict[lidx][inidx])
                weight = float(array[outidx][inidx])
                qm.add_quad(q_ind1=uvaridx, q_ind2=vvaridx, q_val=weight)
    if debug:
        qm.print_info()

    cplexsolve.set_objective(qm, olamb=olamb)
    solution = cplexsolve.get_solution(mode='max')

    # ================================================================================
    # Deploy
    # ================================================================================
    u_dict, v_dict = dict(), dict()
    for lidx, nact in enumerate(nact_list):
        u_dict[lidx] = np.zeros([nact])
        v_dict[lidx] = np.zeros([nact])

        for actidx in range(nact):
            uvaridx = int(u2var_dict[lidx][actidx])
            u_dict[lidx][actidx] = solution[uvaridx]

            vvaridx = int(v2var_dict[lidx][actidx])
            v_dict[lidx][actidx] = solution[vvaridx]

    binary_array_list = [np.zeros_like(array) for array in array_list]
    # array[lidx] => r[lidx] r[lidx+1]
    for lidx, binary in enumerate(binary_array_list):
        nout, nin = np.shape(binary)
        for outidx in range(nout):
            for inidx in range(nin):
                if u_dict[lidx+1][outidx]==1 and v_dict[lidx][inidx]==1 :
                    binary[outidx][inidx]=1

    return u_dict, v_dict

