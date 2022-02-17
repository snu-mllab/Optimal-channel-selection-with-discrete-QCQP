import cplex

class VarManager(object): 
    def __init__(self):
        self.names = list()
        self.lower_bound = list()
        self.upper_bound = list()

    def add(self, name, lower, upper):
        self.names.append(name)
        self.lower_bound.append(lower)
        self.upper_bound.append(upper)

    def get_nvar(self):
        return len(self.names)

def remove_overuse(ind, val):
    ind2val = dict()

    for x, y in zip(ind, val):
        if x not in ind2val.keys():
            ind2val[x] = y
        else:
            ind2val[x] += y

    new_ind = list()
    new_val = list()

    for x, y in ind2val.items():
        new_ind.append(x)
        new_val.append(y)

    return new_ind, new_val

def remove_overuse_quad(q_ind1, q_ind2, q_val):
    ind2val = dict()

    for idx1, idx2, val in zip(q_ind1, q_ind2, q_val):
        key = "{}_{}".format(min(idx1,idx2), max(idx1, idx2))

        if key not in ind2val.keys():
            ind2val[key] = val
        else:
            ind2val[key] += val

    new_q_ind1 = list()
    new_q_ind2 = list()
    new_q_val = list()

    for key, val in ind2val.items():
        idx1, idx2 = key.split('_')
        idx1 = int(idx1)
        idx2 = int(idx2)

        new_q_ind1.append(idx1)
        new_q_ind2.append(idx2)
        new_q_val.append(val)

    return new_q_ind1, new_q_ind2, new_q_val

class LinearManager:
    def __init__(self):
        self.ind = list()
        self.val = list()

    def add(self, ind, val):
        '''
        Args:
            ind - varaible indes
            val - constant before the variable
        '''
        self.ind.append(ind)
        self.val.append(val)

class LinearConstraintsManager:
    def __init__(self):
        self.lin_expr = list()
        self.rhs = list()
        self.names = list()
        self.senses= list()

    def add(self, ind, val, rhs, sense, name):
        '''
        Args:
            ind
            val - float or int
            rhs - float or int
                right hand side value
            sense - "L", "E", "G"
                L -> less than rhs
                E -> equal with rhs
                G -> greater than rhs
            name - string
        '''
        ind, val = remove_overuse(ind=ind, val=val)
        self.lin_expr.append(cplex.SparsePair(ind=ind, val=val))
        self.rhs.append(rhs)
        self.senses.append(sense)
        self.names.append(name)

class QuadraticManager:
    def __init__(self):
        self.l_val = list()
        self.l_ind = list()
        self.q_ind1 = list()
        self.q_ind2 = list()
        self.q_val = list()

    def add_quad(self, q_ind1, q_ind2, q_val):
        '''
        Args:
            q_ind1 - float
            q_ind2 - float
            q_val - float
        '''
        self.q_ind1.append(q_ind1)
        self.q_ind2.append(q_ind2)
        self.q_val.append(q_val)

    def add_linear(self, l_ind, l_val):
        '''
        Args:
            q_ind1 - float
            q_ind2 - float
            q_val - float
        '''
        self.l_ind.append(l_ind)
        self.l_val.append(l_val)
    
    def print_info(self):
        print("l_val : {}".format(len(self.l_val)))
        print("l_ind : {}".format(len(self.l_ind)))
        print("q_ind1 : {}".format(len(self.q_ind1)))
        print("q_ind2 : {}".format(len(self.q_ind2)))
        print("q_val : {}".format(len(self.q_val)))

class CplexSolver(object):
    def __init__(self, timelimit=None, emph_memory=False, file_set=False, variable_select=False, node_select=False):
        self.problem = cplex.Cplex()
        if timelimit is not None:
            print("timelimit is {}".format(timelimit))
            self.problem.parameters.timelimit.set(timelimit)
        if emph_memory:
            self.problem.parameters.emphasis.memory.set(1)
        if file_set:
            self.problem.parameters.mip.strategy.file.set(3)
        if variable_select:
            self.problem.parameters.mip.strategy.variableselect.set(3)
        if node_select:
            self.problem.parameters.mip.strategy.nodeselect.set(0)

    def set_variable(self,  vm):
        '''
        Args:
            vm - VarManager
        '''
        names = vm.names
        lower_bound = vm.lower_bound
        upper_bound = vm.upper_bound

        self.nvar = len(names)

        print("Total {} variables".format(self.nvar))
        self.problem.variables.add(names=names)
        for vidx in range(self.nvar):
            self.problem.variables.set_lower_bounds(vidx, lower_bound[vidx])
            self.problem.variables.set_upper_bounds(vidx, upper_bound[vidx])
            self.problem.variables.set_types(vidx, self.problem.variables.type.integer)

    def set_linear_constraints(self, lcm):
        '''
        Args:
            lcm - LinearConstraintsManager
        '''
        self.problem.linear_constraints.add(lin_expr=lcm.lin_expr, rhs=lcm.rhs, names = lcm.names, senses = lcm.senses)

    def set_quadratic_constraint(self, qm, rhs, sense, name, clamb=None):
        '''
        Args:
            qm - QuadraticManager
            rhs - float
            sense - "L", "E", "G"
                L -> less than rhs
                E -> equal with rhs
                G -> greater than rhs
            name - string
                "quadratic_constraint",
            clamb - float
        '''
        l_ind = qm.l_ind
        l_val = qm.l_val

        q_ind1, q_ind2, q_val = remove_overuse_quad(q_ind1=qm.q_ind1, q_ind2=qm.q_ind2, q_val=qm.q_val)

        ind_set = list(set(q_ind1+q_ind2))
        if clamb is not None:
            for idx in ind_set:
                idx = int(idx)
                q_ind1.append(idx)
                q_ind2.append(idx)
                q_val.append(clamb)

                l_ind.append(idx)
                l_val.append(-clamb)

        l_ind, l_val = remove_overuse(ind=l_ind, val=l_val)

        if len(l_ind)==0:
            self.problem.quadratic_constraints.add(
                    name = name, 
                    quad_expr = cplex.SparseTriple(ind1 = q_ind1, ind2 = q_ind2, val = q_val),
                    rhs = rhs,
                    sense = sense) # Less, Greater, Equal
        else:
            self.problem.quadratic_constraints.add(
                    name = name, 
                    lin_expr = cplex.SparsePair(ind=l_ind, val=l_val),
                    quad_expr = cplex.SparseTriple(ind1 = q_ind1, ind2 = q_ind2, val = q_val),
                    rhs = rhs,
                    sense = sense) # Less, Greater, Equal

    def set_objective(self, qm, olamb=None):
        qmat_ind = [list() for varidx in range(self.nvar)]
        qmat_val = [list() for varidx in range(self.nvar)]

        l_ind = qm.l_ind
        l_val = qm.l_val
        q_ind1 = qm.q_ind1
        q_ind2 = qm.q_ind2
        q_val = qm.q_val

        ind_set = list(set(q_ind1+q_ind2))

        if olamb is not None:
            for idx in ind_set:
                idx = int(idx)

                q_ind1.append(idx)
                q_ind2.append(idx)
                q_val.append(-olamb)
                l_ind.append(idx)
                l_val.append(olamb)

        q_ind1, q_ind2, q_val = remove_overuse_quad(q_ind1=q_ind1, q_ind2=q_ind2, q_val=q_val)
        l_ind, l_val = remove_overuse(ind=l_ind, val=l_val)

        for idx1, idx2, val in zip(q_ind1, q_ind2, q_val):
            if idx1==idx2:
                qmat_ind[idx1].append(idx1)
                qmat_val[idx1].append(2*val)
            else:
                qmat_ind[idx1].append(idx2)
                qmat_val[idx1].append(val)

                qmat_ind[idx2].append(idx1)
                qmat_val[idx2].append(val)

        if len(l_ind)!=0:
            linear_list = list()
            for x, y in zip(l_ind, l_val):
                linear_tmp = (x, y)
                linear_list.append(linear_tmp)
            self.problem.objective.set_linear(linear_list)

        if len(q_ind1)!=0:
            qmat = [cplex.SparsePair(ind=ind, val=val) for ind, val in zip(qmat_ind, qmat_val)]
            self.problem.objective.set_quadratic(qmat)

    def get_solution(self, mode='max'):
        '''
        Args:
            mode - 'max' or 'min'
                'max' => maximize
                'min' => minimize
        '''

        if mode=='max': self.problem.objective.set_sense(self.problem.objective.sense.maximize)
        elif mode=='min': self.problem.objective.set_sense(self.problem.objective.sense.minimize)
        else: assert False, "Wrong mode({})".format(mode)

        self.problem.solve()

        return [int(i) for i in self.problem.solution.get_values()]

def seq_linear_const(linear_consman, nact_list, u2var_dict, v2var_dict, skip_s_list, skip_t_list, is_imagenet):
    #\sum_j u^{(l)}_j >=1, \sum_j v^{(l)}_j >=1
    for lidx, nact in enumerate(nact_list):
        #\sum_j u^{(l)}_j >=1 
        lm = LinearManager()
        for uvaridx in u2var_dict[lidx]:
            lm.add(ind=int(uvaridx), val=1)
        # >=1
        linear_consman.add(ind=lm.ind, val=lm.val, rhs=1, sense="G", name="u_constraint_{}".format(lidx))

        #\sum_j v^{(l)}_j 
        lm = LinearManager()
        for vvaridx in v2var_dict[lidx]:
            lm.add(ind=int(vvaridx), val=1)
        # >=1
        linear_consman.add(ind=lm.ind, val=lm.val, rhs=1, sense="G", name="v_constraint_{}".format(lidx))

    for sidx, tidx in zip(skip_s_list, skip_t_list):
        nact_s = nact_list[sidx]
        nact_t = nact_list[tidx]
        # ImageNet
        if is_imagenet:
            # if downsampling in imagenet
            if nact_s < nact_t:
                for actidx in range(nact_t):
                    utvaridx = int(u2var_dict[tidx][actidx])
                    vtvaridx = int(v2var_dict[tidx][actidx])

                    # u^{t}_j - v^{t}_j <= 0
                    ind = [utvaridx, vtvaridx]
                    val = [1, -1]
                    # <= 0
                    linear_consman.add(ind=ind, val=val, rhs=0, sense="L", name="uv_constraint_{}_{}_down".format(lidx,actidx))
            # if not downsampling in imagenet
            # nact_s = nact_t
            else:
                for actidx in range(nact_t):
                    utvaridx = int(u2var_dict[tidx][actidx])
                    vtvaridx = int(v2var_dict[tidx][actidx])
                    vsvaridx = int(v2var_dict[sidx][actidx])
                    
                    #v^{t}_j - u^{t}_j - v^{s}_j <= 0
                    ind = [vtvaridx, utvaridx, vsvaridx]
                    val = [1, -1, -1]
                    
                    linear_consman.add(ind=ind, val=val, rhs=0, sense="L", name="uv_constraint_{}_{}_1".format(lidx, actidx))

                    #u^{t}_j - v^{t}_j <= 0
                    ind = [utvaridx, vtvaridx]
                    val = [1, -1]

                    # <=0
                    linear_consman.add(ind=ind, val=val, rhs=0, sense="L", name="uv_constraint_{}_{}_2".format(lidx, actidx))
        # Cifar
        else:
            for actidx in range(nact_t):
                utvaridx = int(u2var_dict[tidx][actidx])
                vtvaridx = int(v2var_dict[tidx][actidx])

                if actidx<nact_s:
                    vsvaridx = int(v2var_dict[sidx][actidx])
                    
                    #v^{t}_j - u^{t}_j - v^{s}_j <= 0
                    ind = [vtvaridx, utvaridx, vsvaridx]
                    val = [1, -1, -1]
                    # <=0
                    linear_consman.add(ind=ind, val=val, rhs=0, sense="L", name="uv_constraint_{}_{}_1".format(lidx, actidx))

                    #u^{t}_j - v^{t}_j <= 0
                    ind = [utvaridx, vtvaridx]
                    val = [1, -1]
                    # <=0
                    linear_consman.add(ind=ind, val=val, rhs=0, sense="L", name="uv_constraint_{}_{}_2".format(lidx, actidx))
                else:
                    # v^{t}_j - u^{t}_j
                    ind = [vtvaridx, utvaridx]
                    val = [1, -1]
                    # =0
                    linear_consman.add(ind=ind, val=val, rhs=0, sense="E", name="uv_constraint_{}_{}".format(lidx, actidx))


