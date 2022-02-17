import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from utils.qcqp_support_op import get_nact_list
from utils.cplex_support import VarManager, LinearManager, LinearConstraintsManager, QuadraticManager, CplexSolver  
import numpy as np

def simple_quadratic_4D_skip_solver(
        array_list,
        quadratic_val_list,
        skip_s_list,
        skip_t_list,
        max_constraint,
        u_dict=None,
        v_dict=None,
        timelimit=None,
        emph_memory=True,
        file_set=False,
        variable_select=False,
        node_select=True):
    '''
    ImageNet Resnet
    Args:
        array_list - list of 4D array [L]
        quadratic_val_list - list of int, [L]
        skip_s_list - list of int [N] N<=L
        skip_t_list - list of int [N] N<=L
        max_constraint - int
    '''
    L = len(array_list)
    nact_list = get_nact_list(array_list)
    print("nact_list : {}".format(nact_list))

    cplexsolve = CplexSolver(timelimit=timelimit, emph_memory=emph_memory, file_set=file_set, variable_select=variable_select, node_select=node_select)

    # Set Variables
    vm = VarManager()

    var_idx = -1
    q2var_dict = dict() 

    # q^(l)_j,a,b
    for lidx, array in enumerate(array_list):
        nout, nin, height, width = array.shape
        q2var_dict[lidx] = np.zeros([nout, height, width], dtype=np.int32)
        for outidx in range(nout): 
            for hidx in range(height):
                for widx in range(width):
                    vm.add(name="q{}_{}_{}_{}".format(lidx,outidx,hidx,widx), lower=0.0, upper=1.0)
                    var_idx+=1
                    q2var_dict[lidx][outidx][hidx][widx] = var_idx
    cplexsolve.set_variable(vm)
    # ================================================================================ #
    # Linear constraint
    # ================================================================================ #
    linear_consman = LinearConstraintsManager()

    for lidx, array in enumerate(array_list):
        nout, nin, height, width = array.shape
        for outidx in range(nout): 
            # q^{(l)}_{j,a,b} - e^{(l)}_j <= 0
            for outidx in range(nout): 
                for hidx in range(height):
                    for widx in range(width):
                        qvaridx = int(q2var_dict[lidx][outidx][hidx][widx])

                        ulvar = u_dict[lidx+1][outidx]

                        # q^{(l)}_{j,a,b} 
                        ind = [qvaridx]
                        val = [1]

                        #<=e^{(l)}_j
                        linear_consman.add(ind=ind, val=val, rhs=ulvar, sense="L", name="qu_constraint_{}_{}_{}_{}".format(lidx, outidx, hidx, widx))

            # \sum_{a,b} q^{(l)}_{j,a,b} >= e^{(l)}_j 
            for outidx in range(nout): 
                # e^{(l)}_j
                ulvar = u_dict[lidx+1][outidx]

                lm = LinearManager()
                # \sum_{a,b} q^{(l)}_{j,a,b}
                for hidx in range(height):
                    for widx in range(width):
                        qvaridx = int(q2var_dict[lidx][outidx][hidx][widx])
                        # q^{(l)}_{j,a,b} 
                        lm.add(ind=qvaridx, val=1)
                # >= e^{(l)}_j
                ulvar = u_dict[lidx+1][outidx]
                linear_consman.add(ind=lm.ind, val=lm.val, rhs=ulvar, sense="G", name="min_qu_constraint_{}_{}".format(lidx, outidx))

    # ==================================================================#
    # Quadratic constraint
    # ==================================================================#
    lm = LinearManager()
    for lidx, array in enumerate(array_list):
        nout, nin, height, width = array.shape
        quadratic_val = quadratic_val_list[lidx]
        for outidx in range(nout): 
            for hidx in range(height):
                for widx in range(width):
                    # \sum_i q^{(l)}_{j,a,b} f^{(l-1)}_i
                    qvaridx = int(q2var_dict[lidx][outidx][hidx][widx])
                    lm.add(ind=qvaridx, val= quadratic_val* np.sum(v_dict[lidx]))

    linear_consman.add(ind=lm.ind, val=lm.val, rhs=max_constraint, sense="L", name="quadratic_constraint")
    cplexsolve.set_linear_constraints(linear_consman)
    # ================================================================================
    # Set objective
    # ================================================================================
    qm = QuadraticManager()
    for lidx, array in enumerate(array_list):
        nout, nin, height, width = array.shape
        for outidx in range(nout): 
            for inidx in range(nin):
                for hidx in range(height):
                    for widx in range(width):
                        qvaridx = int(q2var_dict[lidx][outidx][hidx][widx])
                        if v_dict[lidx][inidx]==1:
                            #q^{(l)}_{j,a,b} f^{(l-1)}_i
                            weight = float(array[outidx][inidx][hidx][widx])
                            qm.add_linear(l_ind=qvaridx, l_val=weight)

    cplexsolve.set_objective(qm)
    solution = cplexsolve.get_solution(mode='max')

    q_dict = dict()
    for lidx, array in enumerate(array_list):
        nout, nin, height, width = array.shape
        q_dict[lidx] = np.zeros([nout, height,width], dtype=np.int32)
        for outidx in range(nout): 
            for hidx in range(height):
                for widx in range(width):
                    varidx = int(q2var_dict[lidx][outidx][hidx][widx])
                    q_dict[lidx][outidx][hidx][widx] = solution[varidx]

    binary_array_list = [np.zeros_like(array) for array in array_list]
    # array[lidx] => r[lidx] r[lidx+1]

    for lidx, binary in enumerate(binary_array_list):
        nout, nin, height, width = np.shape(binary)
        for outidx in range(nout):
            for inidx in range(nin):
                for hidx in range(height):
                    for widx in range(width):
                        if u_dict[lidx+1][outidx]==1 and v_dict[lidx][inidx]==1 and q_dict[lidx][outidx][hidx][widx]==1:
                            binary[outidx][inidx][hidx][widx]=1

    return q_dict, binary_array_list

def fast_quadratic_4D_skip_solver(
        array_list,
        quadratic_val_list,
        skip_s_list,
        skip_t_list,
        max_constraint,
        solver,
        gamma=0.5,
        olamb=10.0,
        clamb=10.0,
        keep_idx = [0],
        timelimit=None,
        debug=False):
    '''
    more fast
    L - nlayer 
    Args:
        array_list - list of 4D array [L]
            initial max_nact_list
        quadratic_val_list - list of int, [L]
        skip_s_list - list of int [N] N<=L
        skip_t_list - list of int [N] N<=L
        max_constraint - int
        solver - func
                    solver(
                            array_list,
                            quadratic_val_list,
                            skip_s_list,
                            skip_t_list,
                            max_constraint,
                            olamb,
                            clamb,
                            timelimit, debug)

            
        gamma - to control the raito of 1 in filter 
        olamb - float lambda for objective 
        clamb - float lambda for constraint
        otype - str
            type of output
    '''
    L = len(array_list)
    nact_list = get_nact_list(array_list)

    if debug: print("nact_list : {}".format(nact_list))

    new_quadratic_val_list = list()
    
    for lidx, array in enumerate(array_list):
        nout, nin, height, width = array.shape
        quadratic_val = quadratic_val_list[lidx]
        new_quadratic_val_list.append(height*width*quadratic_val)
    
    new_array_list = [np.sum(array, axis=(2,3)) for array in array_list]
    
    u_dict, v_dict = solver(
                            array_list=new_array_list, quadratic_val_list=new_quadratic_val_list,
                            skip_s_list= skip_s_list, skip_t_list= skip_t_list,
                            max_constraint=max_constraint/gamma,
                            olamb=olamb, clamb=clamb,
                            timelimit=timelimit,
                            debug=debug)
    if debug:
        print(u_dict, v_dict)

    q_dict, binary_array_list = simple_quadratic_4D_skip_solver(
                            array_list=array_list, quadratic_val_list=quadratic_val_list,
                            skip_s_list= skip_s_list, skip_t_list= skip_t_list,
                            max_constraint=max_constraint,
                            u_dict=u_dict, v_dict=v_dict,
                            timelimit=timelimit)
    return u_dict, v_dict, q_dict, binary_array_list

