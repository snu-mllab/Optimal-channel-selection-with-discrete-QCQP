import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.qcqp_support_op import get_nact_list
from utils.write_op import write_pkl
from utils.read_op import read_pkl

import numpy as np
import copy

def get_build_key(lidx, actidx, type_):
    return '{}_{}_{}'.format(type_, lidx, actidx)

def key2item(key):
    type_, lidx, actidx = key.split('_')

    lidx = int(lidx)
    actidx = int(actidx)

    return lidx, actidx, type_

def get_obj_list_4d(array_list, u_dict, v_dict, q_dict=None):
    '''
    Args:
        v_dict - input_dict
        u_dict - output_dict
        q_dict - output wise kernel shape
    '''

    if q_dict is None:
        array_list = [np.sum(array, axis=(2,3)) for array in array_list] 
        return get_obj_list_2d(array_list, u_dict, v_dict)

    obj_list = list()
    for lidx, array in enumerate(array_list):
        obj = 0 
        nout, nin, height, width = np.shape(array)
        for outidx in range(nout):
            for inidx in range(nin):
                for hidx in range(height):
                    for widx in range(width):
                        if u_dict[lidx+1][outidx]==1 and v_dict[lidx][inidx]==1 and q_dict[lidx][outidx][hidx][widx]==1:
                            obj += array[outidx][inidx][hidx][widx]
        obj_list.append(obj)
    return obj_list

def get_obj_list_2d(array_list, u_dict, v_dict): 
    '''
    Args:
        v_dict - input_dict
        u_dict - output_dict
    '''
    obj_list = list()
    for lidx, array in enumerate(array_list):
        obj = 0 
        nout, nin = np.shape(array)
        for outidx in range(nout):
            for inidx in range(nin):
                if u_dict[lidx+1][outidx]==1 and v_dict[lidx][inidx]==1: 
                    obj += array[outidx][inidx]
        obj_list.append(obj)
    return obj_list

def get_cons_list_4d(val_list, vcount_list, qcount_list): 
    cons_list = list()
    for lidx, val in enumerate(val_list):
        nin = vcount_list[lidx]
        noutq = qcount_list[lidx]
        cons_list.append(noutq*nin*val)
    return cons_list 

def get_cons_list_2d(val_list, ucount_list, vcount_list): 
    cons_list = list()
    for lidx, val in enumerate(val_list):
        nout = ucount_list[lidx+1]
        nin = vcount_list[lidx]
        cons_list.append(nout*nin*val)
    return cons_list 

def np_argsort_2d(array):
    nout, nin = array.shape

    var_dict = np.zeros([nout, nin])
    varidx2pos = dict()
    varidx = 0
    val_array = np.zeros([nout*nin])

    for outidx in range(nout):
        for inidx in range(nin):
            var_dict[outidx][inidx] = varidx
            varidx2pos[varidx] = (outidx, inidx)
            val_array[varidx] = array[outidx][inidx]
            varidx+=1

    return [varidx2pos[varidx] for varidx in  np.argsort(val_array)]

class FindManagerPlugin:
    def __init__(self):
        self.value_list = list()
        self.data_list = list()
        self.idx = -1
        self.max_idx = -1
        self.max_value = None
        self.min_idx = -1
        self.min_value = None

    @property
    def descend_order(self):
        return np.argsort(-np.array(self.value_list))

    @property
    def ascend_order(self):
        return np.argsort(np.array(self.value_list))

    def add_item(self, value, data): 
        self.value_list.append(value)
        self.data_list.append(data)
        self.idx += 1
        if self.max_idx==-1:
            self.max_value=value
            self.max_idx=self.idx
        if self.min_idx==-1:
            self.min_value = value
            self.min_idx = self.idx
        if self.max_value<value:
            self.max_value=value
            self.max_idx=self.idx
        if self.min_value>value:
            self.min_value=value
            self.min_idx=self.idx

    def find_max(self):
        return self.data_list[self.max_idx]

    def find_min(self):
        return self.data_list[self.min_idx]

    def isempty(self):
        if len(self.value_list)==0:
            return True
        return False
    
class FindManager(FindManagerPlugin):
    def __init__(self):
        super(FindManager, self).__init__()

    @property
    def nitem(self):
        return self.idx+1

    def add(self, lidx, actidx, value, obj, cons, type_, depend=None):
        data = {'lidx' : lidx, 
                'actidx' : actidx ,
                'obj' : obj,
                'cons' : cons,
                'type' : type_,
                'depend' : depend
                }
        self.add_item(value, data)

    def get_data(self, idx):
        data = self.data_list[idx]
        lidx = data['lidx']
        actidx = data['actidx']
        type_ = data['type']
        obj = data['obj']
        cons = data['cons']
        depend = data['depend']

        if depend is None: 
            return lidx, actidx, obj, cons, type_
        else:
            return lidx, actidx, obj, cons, type_, depend

    def print_data(self):
        for data in self.data_list:
            lidx = data['lidx']
            actidx = data['actidx']
            type_ = data['type']
            print(get_build_key(lidx, actidx, type_))

    def find_max(self):
        return self.get_data(self.max_idx)

    def find_min(self):
        return self.get_data(self.min_idx)

    def find_max_list(self, top_k):
        return [self.get_data(idx) for idx in self.descend_order[:top_k]]

    def find_min_list(self, top_k):
        return [self.get_data(idx) for idx in self.ascend_order[:top_k]]

class DependManager:
    def __init__(self):
        self.depend = dict()

    @property
    def nitem_dict(self):
        return {key : len(self.depend[key]) for key in self.depend.keys()}

    def add_key(self, key):
        self.depend[key] = list()

    def get_item(self, key, idx):
        content = self.depend[key][idx] 
        lidx = content['lidx']
        actidx = content['actidx']
        type_ = content['type'] 
        
        return lidx, actidx, type_

    def get_depend_key_list(self, key):
        nitem = self.nitem_dict[key]

        key_list = list()
        for idx in range(nitem):
            content = self.depend[key][idx] 
            lidx = content['lidx']
            actidx = content['actidx']
            type_ = content['type'] 
            key_list.append(get_build_key(lidx, actidx, type_))
        return key_list

    def append_item(self, key, lidx, actidx, type_):
        if key not in self.depend.keys(): self.add_key(key)
        self.depend[key].append({'lidx' : lidx, 'actidx' : actidx, 'type' : type_})

    def append_items(self, source_key, target_key):
        if target_key not in self.depend.keys(): self.add_key(target_key)
        if source_key not in self.depend.keys(): self.add_key(source_key)
        for item in self.depend[source_key]:
            self.depend[target_key].append(copy.deepcopy(item))

class Var2DManager:
    def __init__(self, array_list, logfunc=print, is_imagenet=False):
        self.array_list = array_list
        self.nact_list = get_nact_list(self.array_list)
        self.logfunc=logfunc
        self.r_dict=None
        self.u_dict=None
        self.v_dict=None

        self.skip_s_list = list()
        self.skip_t_list = list()
        self.is_imagenet = is_imagenet

    def set_val_list(self, val_list):
        self.val_list =val_list

    @property
    def obj_list(self):
        if self.r_dict is not None:
            return get_obj_list_2d(self.array_list, self.r_dict, self.r_dict)
        else:
            return get_obj_list_2d(self.array_list, self.u_dict, self.v_dict)

    @property
    def cons_list(self): 
        if self.r_dict is not None:
            return get_cons_list_2d(self.val_list, self.rcount_list, self.rcount_list)
        else:
            return get_cons_list_2d(self.val_list, self.ucount_list, self.vcount_list)

    @property
    def total_obj(self):
        return sum(self.obj_list)

    @property
    def total_cons(self):
        return sum(self.cons_list)

    @property
    def max_cons(self):
        cons = sum(get_cons_list_2d(self.val_list, self.nact_list, self.nact_list))
        return cons

    @property
    def L(self):
        return  len(self.array_list)

    @property
    def N(self):
        return len(self.skip_s_list)

    @property
    def rcount_list(self):
        return [int(np.sum(self.r_dict[lidx])) for lidx in range(self.L+1)]

    @property
    def ucount_list(self):
        return [int(np.sum(self.u_dict[lidx])) for lidx in range(self.L+1)]

    @property
    def vcount_list(self):
        return [int(np.sum(self.v_dict[lidx])) for lidx in range(self.L+1)]

    def set_min_select_list(self, min_select_list):
        self.min_select_list = min_select_list

    def set_skip_list(self, s_list, t_list):
        self.skip_s_list = s_list
        self.skip_t_list = t_list

        self.s2t_dict = {sidx : tidx for sidx, tidx in zip(self.skip_s_list, self.skip_t_list)}
        self.t2s_dict = {tidx : sidx for sidx, tidx in zip(self.skip_s_list, self.skip_t_list)}

        for sidx, tidx in self.s2t_dict.items(): 
            if tidx>=self.L+1:
                continue
            assert self.nact_list[sidx] <= self.nact_list[tidx], f"Wrong condition self.nact_list[{sidx}](={self.nact_list[sidx]}) > self.nact_list[{tidx}](={self.nact_list[tidx]})" 

    def init_r_dict(self):
        self.r_dict = {lidx : np.zeros([nact]) for lidx, nact in enumerate(self.nact_list)}
        for lidx, nact in enumerate(self.nact_list):
            min_select = self.min_select_list[lidx] 
            for actidx in range(min_select):
                self.r_dict[lidx][actidx] = 1.0

    def init_uv_dict(self):
        self.init_uv_dict_custom(self.r_dict, self.r_dict)
        self.r_dict = None

    def init_r_dict_custom(self, r_dict):
        self.r_dict = copy.deepcopy(r_dict)

    def init_uv_dict_custom(self, u_dict, v_dict):
        self.u_dict = copy.deepcopy(u_dict)
        self.v_dict = copy.deepcopy(v_dict)


    def select_dict(self, type_):
        if type_== 'u':
            return self.u_dict
        if type_== 'v':
            return self.v_dict


    def is_full_custom(self, lidx, actidx, dict_):
        if dict_[lidx][actidx]==1:
            return True
        else:
            return False

    def is_full_u(self, lidx, actidx):
        return self.is_full_custom(lidx, actidx, self.u_dict)

    def is_full_v(self, lidx, actidx):
        return self.is_full_custom(lidx, actidx, self.v_dict)

    def is_full(self, lidx, actidx, type_):
        if type_=='u':
            return self.is_full_u(lidx, actidx)
        elif type_=='v':
            return self.is_full_v(lidx, actidx)

    def is_empty_custom(self, lidx, actidx, dict_):
        if dict_[lidx][actidx]==0:
            return True
        else:
            return False

    def is_empty_u(self, lidx, actidx):
        return self.is_empty_custom(lidx, actidx, self.u_dict)

    def is_empty_v(self, lidx, actidx):
        return self.is_empty_custom(lidx, actidx, self.v_dict)

    def is_empty(self, lidx, actidx, type_):
        if type_=='u':
            return self.is_empty_u(lidx, actidx)
        elif type_=='v':
            return self.is_empty_v(lidx, actidx)

    def fill_custom(self, lidx, actidx, dict_):
        dict_[lidx][actidx] = 1

    def fill_u(self, lidx, actidx):
        self.fill_custom(lidx, actidx, self.u_dict)
        if lidx>0:
            lidx2 = lidx-1
            type2 = 'v'

            array = self.array_list[lidx-1]
            val = self.val_list[lidx-1]
            nact = self.nact_list[lidx-1]

            for actidx2 in range(nact):
                key2 = get_build_key(lidx2, actidx2, type2)
                self.subcal_save_dict[key2]['obj'] += array[actidx][actidx2]
                self.subcal_save_dict[key2]['cons'] += val 


    def fill_v(self, lidx, actidx):
        self.fill_custom(lidx, actidx, self.v_dict)
        if lidx+1<=self.L:
            lidx2 = lidx+1
            type2 = 'u'

            array = self.array_list[lidx]
            val = self.val_list[lidx]
            nact = self.nact_list[lidx2]

            for actidx2 in range(nact):
                key2 = get_build_key(lidx2, actidx2, type2)
                self.subcal_save_dict[key2]['obj'] += array[actidx2][actidx]
                self.subcal_save_dict[key2]['cons'] += val 

    def fill(self, lidx, actidx, type_):
        if type_=='u':
            self.fill_u(lidx, actidx)
        elif type_=='v':
            self.fill_v(lidx, actidx)

    def remove_custom(self, lidx, actidx, dict_):
        dict_[lidx][actidx] = 0

    def remove_u(self, lidx, actidx):
        self.remove_custom(lidx, actidx, self.u_dict)

    def remove_v(self, lidx, actidx):
        self.remove_custom(lidx, actidx, self.v_dict)

    def remove(self, lidx, actidx, type_):
        if type_=='u':
            return self.remove_u(lidx, actidx)
        elif type_=='v':
            return self.remove_v(lidx, actidx)

    def save(self, savepath):
        if self.r_dict is not None: 
            r_dict, binary_array_list = self.get_solution()
            content = {
                    'r_dict' : r_dict,
                    'binary_array_list' : binary_array_list
                    }
            write_pkl(content=content, path=savepath) 
        else:
            u_dict, v_dict, binary_array_list = self.get_solution()
            content = {
                    'u_dict' : u_dict,
                    'v_dict' : v_dict,
                    'binary_array_list' : binary_array_list
                    }
            write_pkl(content=content, path=savepath) 

    def load(self, loadpath):
        content = read_pkl(load_path)
        if 'r_dict' in content.keys():
            r_dict = content['r_dict']
            binary_array_list = content['binary_array_list']
            self.init_r_dict_custom(r_dict)
        else:
            u_dict = content['u_dict']
            v_dict = content['v_dict']
            binary_array_list = content['binary_array_list']
            self.init_uv_dict_custom(u_dict, v_dict)

    def get_solution(self):
        binary_array_list = [np.zeros_like(array) for array in self.array_list]
        for lidx, binary in enumerate(binary_array_list):
            nout, nin = np.shape(binary)
            for outidx in range(nout):
                for inidx in range(nin):
                    if self.r_dict is not None:
                        if self.r_dict[lidx+1][outidx]==1 and self.r_dict[lidx][inidx]==1:
                            binary[outidx][inidx]=1
                    else:
                        if self.u_dict[lidx+1][outidx]==1 and self.v_dict[lidx][inidx]==1:
                            binary[outidx][inidx]=1

        if self.r_dict is not None:
            return self.r_dict, binary_array_list
        else:
            return self.u_dict, self.v_dict, binary_array_list

    def get_divide_opt_block(self, divide=3, left=0):
        opt_block = [idx  for idx in range(self.L+1) if idx%divide==left] 
        return opt_block 

    def simple_optimize(self, opt_block, nselect_list, r_dict, array_list=None):
        if array_list is None:
            array_list = self.array_list

        for lidx, nselect in enumerate(nselect_list):
            nact = self.nact_list[lidx]
            val_array = np.zeros([nact])
            if lidx in opt_block:
                # case 1 
                if lidx-1 not in opt_block and lidx>0:
                    array = array_list[lidx-1]
                    prev_nact = self.nact_list[lidx-1]
                    for actidx in range(nact):
                        val_tmp = 0 
                        for prev_actidx in range(prev_nact): 
                            weight = float(array[actidx][prev_actidx])
                            val_tmp+= weight*r_dict[lidx-1][prev_actidx]
                        val_array[actidx] += val_tmp

                if lidx+1 not in opt_block and lidx < self.L:
                    array = array_list[lidx]
                    post_nact = self.nact_list[lidx+1]
                    for actidx in range(nact):
                        val_tmp = 0
                        for post_actidx in range(post_nact): 
                            weight = float(array[post_actidx][actidx])
                            val_tmp += weight*r_dict[lidx+1][post_actidx]
                        val_array[actidx] += val_tmp 

                r_dict[lidx] = np.zeros([nact])
                for actidx in np.argsort(-val_array)[:nselect]:
                    r_dict[lidx][actidx] = 1

    def optimize_iterate(self, divide=2):
        cur_obj = self.total_obj
        while True:
            for left in range(divide):
                opt_block = self.get_divide_opt_block(divide=divide, left=left)
                nselect_list = copy.deepcopy(self.min_select_list)
                self.simple_optimize(opt_block=opt_block, nselect_list=nselect_list, r_dict=self.r_dict)
                obj = self.total_obj
                self.logfunc(" obj : {}".format(obj))
            if obj==cur_obj:
                break
            cur_obj = obj

    def cal_obj_cons_add_case1(self, lidx, actidx, type_):
        key = get_build_key(lidx, actidx, type_)
        p_obj = self.subcal_save_dict[key]['obj']
        p_cons = self.subcal_save_dict[key]['cons']
        p_depend = list()

        if type_=='u':
            type2='v'
        else:
            type2='u'
        key2 = get_build_key(lidx, actidx, type2)

        if not self.is_full(lidx, actidx, type2):
            p_obj += self.subcal_save_dict[key2]['obj']
            p_cons += self.subcal_save_dict[key2]['cons']
            p_depend.append({'lidx' : lidx, 'actidx' : actidx, 'type' : type2})

        return p_obj, p_cons, p_depend

    def cal_obj_cons_add_case2(self, lidx, actidx, type_):
        key = get_build_key(lidx, actidx, type_)
        p_obj = self.subcal_save_dict[key]['obj']
        p_cons = self.subcal_save_dict[key]['cons']
        p_depend = list()

        if type_=='u':
            type2='v'
            key2 = get_build_key(lidx, actidx, type2)
            if not self.is_full(lidx, actidx, type2):
                p_obj += self.subcal_save_dict[key2]['obj']
                p_cons += self.subcal_save_dict[key2]['cons']
                p_depend.append({'lidx' : lidx, 'actidx' : actidx, 'type' : type2})
            return p_obj, p_cons, p_depend

        cur_obj = p_obj
        cur_cons = p_cons
        cur_key_list = [key]

        fm = FindManagerPlugin()

        tidx = lidx
        sidx = self.t2s_dict[tidx]
        nact_t = self.nact_list[tidx]
        nact_s = self.nact_list[sidx]

        while True:
            if nact_t==nact_s and self.is_empty(sidx, actidx, 'v'):
                ukey = get_build_key(tidx, actidx, 'u')

                obj = cur_obj + self.subcal_save_dict[ukey]['obj'] 
                cons = cur_cons + self.subcal_save_dict[ukey]['cons'] 
                key_list =  cur_key_list + [ukey]

                if cons==0:
                    fm.add_item(0, {'key':key_list,  'obj':obj, 'cons':cons}  )
                else:
                    fm.add_item(obj/cons, {'key':key_list,  'obj':obj, 'cons':cons}  )

                vkey = get_build_key(sidx, actidx, 'v')
                cur_obj += self.subcal_save_dict[vkey]['obj']
                cur_cons += self.subcal_save_dict[vkey]['cons']
                cur_key_list.append(vkey)

            else: 
                if cur_cons==0:
                    fm.add_item(0, {'key' : cur_key_list,  'obj' : cur_obj, 'cons':cur_cons})
                else:
                    fm.add_item(cur_obj/cur_cons, {'key' : cur_key_list,  'obj' : cur_obj, 'cons':cur_cons} )
                break

            if sidx not in self.skip_t_list:
                ukey = get_build_key(sidx, actidx, 'u')
                obj = cur_obj + self.subcal_save_dict[ukey]['obj'] 
                cons = cur_cons + self.subcal_save_dict[ukey]['cons'] 
                key_list =  cur_key_list + [ukey]
                if cons==0:
                    fm.add_item(0, {'key':key_list,  'obj':obj, 'cons':cons}  )
                else:
                    fm.add_item(obj/cons, {'key':key_list,  'obj':obj, 'cons':cons}  )
                break
            tidx = sidx
            sidx = self.t2s_dict[tidx]
            nact_t = self.nact_list[tidx]
            nact_s = self.nact_list[sidx]

        max_item = fm.find_max()

        key_list = max_item['key']
        p_obj = max_item['obj']
        p_cons = max_item['cons']

        p_depend = list()

        for key in key_list:
            lidx, actidx, type_ = key2item(key)
            p_depend.append({'lidx' : lidx, 'actidx' : actidx, 'type' : type_})

        return p_obj, p_cons, p_depend

    def cal_obj_cons_add_case3(self, lidx, actidx, type_):
        key = get_build_key(lidx, actidx, type_)
        p_obj = self.subcal_save_dict[key]['obj']
        p_cons = self.subcal_save_dict[key]['cons']
        p_depend = list()

        if type_=='u':
            type2='v'
            key2 = get_build_key(lidx, actidx, type2)
            if not self.is_full(lidx, actidx, type2):
                p_obj += self.subcal_save_dict[key2]['obj']
                p_cons += self.subcal_save_dict[key2]['cons']
                p_depend.append({'lidx' : lidx, 'actidx' : actidx, 'type' : type2})
            return p_obj, p_cons, p_depend

        cur_obj = p_obj
        cur_cons = p_cons
        cur_key_list = []

        fm = FindManagerPlugin()

        tidx = lidx
        sidx = self.t2s_dict[tidx]
        nact_t = self.nact_list[tidx]
        nact_s = self.nact_list[sidx]

        while True:
            if actidx<nact_s: 
                if self.is_empty(sidx, actidx, 'v'):
                    ukey = get_build_key(tidx, actidx, 'u')

                    obj = cur_obj + self.subcal_save_dict[ukey]['obj'] 
                    cons = cur_cons + self.subcal_save_dict[ukey]['cons'] 
                    key_list =  cur_key_list + [ukey]

                    if cons==0:
                        fm.add_item(0, {'key':key_list,  'obj':obj, 'cons':cons}  )
                    else:
                        fm.add_item(obj/cons, {'key':key_list,  'obj':obj, 'cons':cons}  )

                    vkey = get_build_key(sidx, actidx, 'v')
                    cur_obj += self.subcal_save_dict[vkey]['obj']
                    cur_cons += self.subcal_save_dict[vkey]['cons']
                    cur_key_list.append(vkey)
                else:
                    if cur_cons==0:
                        fm.add_item(0, {'key' : cur_key_list,  'obj' : cur_obj, 'cons':cur_cons})
                    else:
                        fm.add_item(cur_obj/cur_cons, {'key' : cur_key_list,  'obj' : cur_obj, 'cons':cur_cons} )
                    break
            else: 
                ukey = get_build_key(tidx, actidx, 'u')
                obj = cur_obj + self.subcal_save_dict[ukey]['obj'] 
                cons = cur_cons + self.subcal_save_dict[ukey]['cons'] 
                key_list =  cur_key_list + [ukey]
                if cons==0:
                    fm.add_item(0, {'key':key_list,  'obj':obj, 'cons':cons}  )
                else:
                    fm.add_item(obj/cons, {'key':key_list,  'obj':obj, 'cons':cons}  )
                break

            if sidx not in self.skip_t_list:
                ukey = get_build_key(sidx, actidx, 'u')
                obj = cur_obj + self.subcal_save_dict[ukey]['obj'] 
                cons = cur_cons + self.subcal_save_dict[ukey]['cons'] 
                key_list =  cur_key_list + [ukey]
                if cons==0:
                    fm.add_item(0, {'key':key_list,  'obj':obj, 'cons':cons}  )
                else:
                    fm.add_item(obj/cons, {'key':key_list,  'obj':obj, 'cons':cons}  )
                break
            tidx = sidx
            sidx = self.t2s_dict[tidx]
            nact_t = self.nact_list[tidx]
            nact_s = self.nact_list[sidx]

        max_item = fm.find_max()

        key_list = max_item['key']
        p_obj = max_item['obj']
        p_cons = max_item['cons']

        for key in key_list:
            lidx, actidx, type_ = key2item(key)
            p_depend.append({'lidx' : lidx, 'actidx' : actidx, 'type' : type_})

        return p_obj, p_cons, p_depend

    def cal_obj_cons_add(self, lidx, actidx, type_):
        if lidx not in self.skip_t_list:
            return self.cal_obj_cons_add_case1(lidx, actidx, type_)
            
        if lidx in self.skip_t_list: 
            if self.is_imagenet:
                return self.cal_obj_cons_add_case2(lidx, actidx ,type_)
            else:
                return self.cal_obj_cons_add_case3(lidx, actidx ,type_)

    def cal_obj_cons(self, lidx, actidx, type_):
        key = get_build_key(lidx, actidx, type_)

        if not self.is_full(lidx, actidx, type_):
            p_obj, p_cons, depend = self.cal_obj_cons_add(lidx, actidx, type_)
            return p_obj, p_cons, depend

    def check_increase_effect(self, limit):
        fm = FindManager()

        for lidx, nact in enumerate(self.nact_list):
            #self.logfunc(f"Check_increase {lidx}, {nact}")
            for actidx in range(nact):
                if lidx in self.skip_s_list or lidx in self.skip_t_list:
                    if not self.is_full(lidx, actidx, 'u'): 
                        p_obj, p_cons, p_depend = self.cal_obj_cons(lidx, actidx,'u')
                        if limit>=p_cons and p_cons!=0:
                            fm.add(lidx,actidx,p_obj/p_cons, p_obj, p_cons, 'u', p_depend)

                    if not self.is_full(lidx, actidx, 'v'): 
                        p_obj, p_cons, p_depend = self.cal_obj_cons(lidx, actidx,'v')
                        if limit>=p_cons and p_cons!=0:
                            fm.add(lidx,actidx,p_obj/p_cons, p_obj, p_cons, 'v', p_depend)
                else:
                    if not self.is_full(lidx, actidx, 'u'): 
                        p_obj, p_cons, p_depend = self.cal_obj_cons(lidx, actidx,'u')
                        if limit>=p_cons and p_cons!=0:
                            fm.add(lidx,actidx,p_obj/p_cons, p_obj, p_cons, 'u', p_depend)
        return fm

    def activate_p_depend(self, p_depend):
        for item in p_depend:
            lidx2 = item['lidx'] 
            actidx2 = item['actidx']
            type2 = item['type']

            self.fill(lidx2, actidx2, type2)

    def greedy_fill(self, max_cons, step=1):
        while True:
            prev_cons = self.total_cons
            prev_obj = self.total_obj

            res_cons = max_cons - prev_cons
            self.logfunc(f"prev_obj : {prev_obj}, cur_cons : ({prev_cons}/{max_cons})")

            self.logfunc("Investigate start (res_cons = {})".format(res_cons))
            fm = self.check_increase_effect(limit=res_cons)
            self.logfunc("Investigate end")
            self.logfunc(f"FindManager size : {fm.nitem}")

            if fm.isempty():
                self.logfunc("nothing to increase")
                break

            cons_sum = 0
            for lidx, actidx, obj, cons, type_, p_depend in fm.find_max_list(step):
                cons_sum += cons
                if cons_sum>res_cons:
                    cons_sum -= cons
                    continue
                self.fill(lidx, actidx, type_)
                self.activate_p_depend(p_depend)
            
            cur_cons = self.total_cons
            cur_obj = self.total_obj
            self.logfunc(f"obj : {cur_obj}, cur_cons : ({cur_cons}/{max_cons})")

            if prev_cons==cur_cons or cur_cons>max_cons: break

            if step==1:
                assert np.allclose(cur_cons, prev_cons + cons_sum), f"Constraint not matched {cur_cons - prev_cons}, {cons_sum}"
                assert np.allclose(cur_obj, prev_obj + obj), f"Objective not matched {cur_obj}, {prev_obj+obj}"


class Var4DManager(Var2DManager):
    def __init__(self, array_list, logfunc=print, is_imagenet=False):
        super(Var4DManager, self).__init__(array_list, logfunc=logfunc, is_imagenet=is_imagenet)
        self.q_dict=None

    @property
    def obj_list(self):
        if self.r_dict is not None:
            if self.q_dict is not None: 
                return get_obj_list_4d(self.array_list, self.r_dict, self.r_dict, self.q_dict)
            else:
                return get_obj_list_4d(self.array_list, self.r_dict, self.r_dict)
        else:
            if self.q_dict is not None: 
                return get_obj_list_4d(self.array_list, self.u_dict, self.v_dict, self.q_dict)
            else:
                return get_obj_list_4d(self.array_list, self.u_dict, self.v_dict)


    @property
    def cons_list(self): 
        if self.r_dict is not None:
            return get_cons_list_4d(self.val_list, self.rcount_list, self.qcount_list)
        else:
            return get_cons_list_4d(self.val_list, self.vcount_list, self.qcount_list)

    @property
    def qcount_list(self):
        return [int(np.sum(self.q_dict[lidx])) for lidx in range(self.L)]

    @property
    def default_qcount_list(self):
        return [int(np.sum(self.default_q_dict[lidx])) for lidx in range(self.L)]

    @property
    def max_cons(self):
        cons = sum(get_cons_list_4d(self.val_list, self.nact_list, self.default_qcount_list))
        return cons

    def is_full_u(self, lidx, actidx):
        if lidx==0:
            return self.is_full_custom(lidx, actidx, self.u_dict)
        else:
            nout, nin, height, width = self.array_list[lidx-1].shape
            if np.sum(self.q_dict[lidx-1][actidx])< height*width:
                return False
            else:
                return True

    def fill_u(self, lidx, actidx):
        # 'u'
        type_ = 'u'
        key = get_build_key(lidx, actidx, type_)
        self.fill_custom(lidx, actidx, self.u_dict)
        if lidx>0:
            array = self.array_list[lidx-1]
            val = self.val_list[lidx-1]
            nact = self.nact_list[lidx-1]

            lidx2 = lidx-1
            type2 = 'v'

            in_nact = array.shape[1]
            height = array.shape[2]
            width = array.shape[3]

            array2d = self.subcal_save_dict[key]['array2d']
            hidx, widx = np_argsort_2d(-array2d)[0]

            self.q_dict[lidx-1][actidx][hidx][widx]=1
            array2d[hidx][widx] = 0

            self.subcal_save_dict[key]['array2d'] = array2d
            self.subcal_save_dict[key]['obj'] = np.max(array2d)

            for actidx2 in range(nact):
                key2 = get_build_key(lidx2, actidx2, type2)
                self.subcal_save_dict[key2]['obj'] += array[actidx][actidx2][hidx][widx]
                self.subcal_save_dict[key2]['cons'] += val 

    def fill_v(self, lidx, actidx):
        # 'v'
        self.fill_custom(lidx, actidx, self.v_dict)
        if lidx+1<=self.L:
            array = self.array_list[lidx]
            val = self.val_list[lidx]
            lidx2 = lidx+1
            type2 = 'u'
            nact = self.nact_list[lidx2]

            height = array.shape[2]
            width = array.shape[3]

            for actidx2 in range(nact):
                key2 = get_build_key(lidx2, actidx2, type2)
                array2d = self.subcal_save_dict[key2]['array2d']

                for hidx in range(height):
                    for widx in range(width):
                        if self.q_dict[lidx][actidx2][hidx][widx]==0:
                            array2d[hidx][widx] += array[actidx2][actidx][hidx][widx]
                self.subcal_save_dict[key2]['obj'] = np.max(array2d)
                self.subcal_save_dict[key2]['cons'] += val 
                self.subcal_save_dict[key2]['array2d'] = array2d

    def init_q_dict(self):
        self.default_q_dict = dict()
        self.q_dict = dict()

        for lidx, array in enumerate(self.array_list):
            nout, nin, height, width = array.shape
            self.q_dict[lidx] = np.zeros([nout, height, width])
            self.default_q_dict[lidx] = np.ones([nout, height, width])

            array3d = np.zeros([nout, height, width])

            for outidx in range(nout):
                for inidx in range(nin):
                    if self.r_dict[lidx+1][outidx]==1 and self.r_dict[lidx][inidx]==1:
                        for hidx in range(height):
                            for widx in range(width):
                                array3d[outidx][hidx][widx] += array[outidx][inidx][hidx][widx]

            for outidx in range(nout):
                if self.r_dict[lidx+1][outidx] == 1:
                    hidx, widx = np_argsort_2d(-array3d[outidx])[0]
                    self.q_dict[lidx][outidx][hidx][widx] = 1

    def init_q_dict_custom(self, q_dict):
        self.q_dict = copy.deepcopy(q_dict)
        self.default_q_dict = dict()
        for key, value in q_dict.items():
            self.default_q_dict[key] = np.ones_like(value)

    def save(self, savepath):
        if self.r_dict is not None: 
            r_dict, q_dict, binary_array_list = self.get_solution()
            content = {
                    'r_dict' : r_dict,
                    'q_dict' : q_dict,
                    'binary_array_list' : binary_array_list
                    }
            write_pkl(content=content, path=savepath) 
        else:
            u_dict, v_dict, q_dict, binary_array_list = self.get_solution()
            content = {
                    'u_dict' : u_dict,
                    'v_dict' : v_dict,
                    'q_dict' : q_dict,
                    'binary_array_list' : binary_array_list
                    }
            write_pkl(content=content, path=savepath) 

    def load(self, loadpath):
        content = read_pkl(load_path)
        if 'r_dict' in content.keys():
            r_dict = content['r_dict']
            q_dict = content['q_dict']
            binary_array_list = content['binary_array_list']
            self.init_r_dict_custom(r_dict)
            self.init_q_dict_custom(q_dict)
        else:
            u_dict = content['u_dict']
            v_dict = content['v_dict']
            q_dict = content['q_dict']
            binary_array_list = content['binary_array_list']
            self.init_uv_dict_custom(u_dict, v_dict)
            self.init_q_dict_custom(q_dict)

    def get_solution(self):
        binary_array_list = [np.zeros_like(array) for array in self.array_list]
        for lidx, binary in enumerate(binary_array_list):
            nout, nin, height, width = np.shape(binary)
            for outidx in range(nout):
                for inidx in range(nin):
                    for hidx in range(height):
                        for widx in range(width):
                            if self.r_dict is not None:
                                if self.r_dict[lidx+1][outidx]==1 and self.r_dict[lidx][inidx]==1 and self.q_dict[lidx][outidx][hidx][widx]==1:
                                    binary[outidx][inidx][hidx][widx]=1
                            else:
                                if self.u_dict[lidx+1][outidx]==1 and self.v_dict[lidx][inidx]==1 and self.q_dict[lidx][outidx][hidx][widx]==1:
                                    binary[outidx][inidx][hidx][widx]=1

        if self.r_dict is not None:
            return self.r_dict, self.q_dict, binary_array_list
        else:
            return self.u_dict, self.v_dict, self.q_dict, binary_array_list

    def optimize_iterate(self, divide=2):
        cur_obj = self.total_obj

        array_list = list()
        for lidx, array in enumerate(self.array_list): 
            nout, nin, height, width = np.shape(array)
            array2d = np.zeros([nout, nin])
            for outidx in range(nout):
                for inidx in range(nin):
                    array2d[outidx][inidx] = np.sum(array[outidx][inidx])
            array_list.append(array2d)

        while True:
            for left in range(divide):
                opt_block = self.get_divide_opt_block(divide=divide, left=left)
                nselect_list = copy.deepcopy(self.min_select_list)
                self.simple_optimize(opt_block=opt_block, nselect_list=nselect_list, r_dict=self.r_dict, array_list=array_list)
                obj = self.total_obj
                self.logfunc(" obj : {}".format(obj))
            if obj==cur_obj:
                break
            cur_obj = obj

class SkipFastCplexSolver(Var2DManager): 
    def __init__(self,
            array_list,
            min_select_list,
            skip_s_list,
            skip_t_list,
            logfunc=print,
            is_imagenet=False,
            ):
        super(SkipFastCplexSolver, self).__init__(array_list=array_list, logfunc=logfunc, is_imagenet=is_imagenet)
        self.set_min_select_list(min_select_list)
        self.set_skip_list(skip_s_list, skip_t_list)

        self.subcal_save_dict = dict()

    def init_custom(self, divide=2):
        self.logfunc(f"{sum(self.nact_list)}")
        self.init_r_dict()
        self.optimize_iterate(divide=divide)
        self.init_uv_dict()
        self.make_subcal_save_dict()

    def sub_cal_obj_cons_custom(self, actidx, type_, alive, array, val):
        obj, cons = 0, 0
        if type_=='in':
            out_nact = array.shape[0]
            for out_actidx in range(out_nact):
                if alive[out_actidx]==1:
                    obj += array[out_actidx][actidx]
                    cons += val
        elif type_=='out':
            in_nact = array.shape[1]
            for in_actidx in range(in_nact):
                if alive[in_actidx]==1:
                    obj += array[actidx][in_actidx]
                    cons += val
        else:
            assert False, "wrong type_"
        return obj, cons


    def sub_cal_obj_cons_v(self, lidx, actidx):
        if lidx<self.L: 
            array = self.array_list[lidx]
            val = self.val_list[lidx]
            obj, cons = self.sub_cal_obj_cons_custom(actidx, 'in', self.u_dict[lidx+1], array, val)
            return obj, cons
        return 0,0

    def sub_cal_obj_cons_u(self, lidx, actidx):
        if lidx>0:
            array = self.array_list[lidx-1]
            val = self.val_list[lidx-1]
            obj, cons = self.sub_cal_obj_cons_custom(actidx, 'out', self.v_dict[lidx-1], array, val)
            return obj, cons
        return 0,0

    def sub_cal_obj_cons(self, lidx, actidx, type_):
        key = get_build_key(lidx, actidx, type_)

        if key in self.subcal_save_dict.keys():
            obj = self.subcal_save_dict[key]['obj'] 
            cons = self.subcal_save_dict[key]['cons'] 
        else:
            if type_=='u':
                obj, cons = self.sub_cal_obj_cons_u(lidx, actidx)
            elif type_=='v':
                obj, cons = self.sub_cal_obj_cons_v(lidx, actidx)
            self.subcal_save_dict[key] = {'obj' : obj, 'cons' : cons}
        return obj, cons

    def make_subcal_save_dict(self):
        self.logfunc("make subcal start")
        for lidx, nact in enumerate(self.nact_list):
            for actidx in range(nact):
                obj, cons = self.sub_cal_obj_cons_u(lidx, actidx)
                key = get_build_key(lidx, actidx, 'u')
                self.subcal_save_dict[key] = {'obj' : obj, 'cons' : cons}

                obj, cons = self.sub_cal_obj_cons_v(lidx, actidx)
                key = get_build_key(lidx, actidx, 'v')
                self.subcal_save_dict[key] = {'obj' : obj, 'cons' : cons}
        self.logfunc("make subcal end")


class SkipFastCplex4DSolver(Var4DManager): 
    def __init__(self,
            array_list,
            min_select_list,
            skip_s_list,
            skip_t_list,
            logfunc=print,
            is_imagenet=False
            ):
        super(SkipFastCplex4DSolver, self).__init__(array_list=array_list, logfunc=logfunc, is_imagenet=is_imagenet)
        self.set_min_select_list(min_select_list)
        self.set_skip_list(skip_s_list, skip_t_list) 

        self.subcal_save_dict = dict()

    def init_custom(self, divide=2):
        self.logfunc(f"{sum(self.nact_list)}")
        self.init_r_dict()
        self.optimize_iterate(divide=divide)
        self.init_q_dict()
        self.init_uv_dict()
        self.make_subcal_save_dict()

    def sub_cal_obj_cons_custom(self, actidx, type_, alive, array, val, out_q =None):
        obj, cons = 0, 0
        if type_=='in4d':
            out_nact = array.shape[0]
            height = array.shape[2]
            width = array.shape[3]

            for out_actidx in range(out_nact):
                for hidx in range(height):
                    for widx in range(width):
                        if alive[out_actidx][hidx][widx]==1:
                            obj += array[out_actidx][actidx][hidx][widx]
                            cons += val
            return obj, cons
        elif type_=='out4d':
            in_nact = array.shape[1]
            height = array.shape[2]
            width = array.shape[3]

            array2d = np.zeros([height, width])

            for in_actidx in range(in_nact):
                if alive[in_actidx]==1:
                    cons += val
                    for hidx in range(height):
                        for widx in range(width):
                            if out_q[actidx][hidx][widx]==0:
                                array2d[hidx][widx]+= array[actidx][in_actidx][hidx][widx]
            obj = np.max(array2d)
            return obj, cons, array2d
        else:
            assert False, "wrong type_"

    def sub_cal_obj_cons_v(self, lidx, actidx):
        if lidx<self.L: 
            array = self.array_list[lidx]
            val = self.val_list[lidx]
            obj, cons = self.sub_cal_obj_cons_custom(actidx, 'in4d', self.q_dict[lidx], array, val)
            return obj, cons
        return 0,0

    def sub_cal_obj_cons_u(self, lidx, actidx):
        if lidx>0:
            array = self.array_list[lidx-1]
            val = self.val_list[lidx-1]
            obj, cons, array2d  = self.sub_cal_obj_cons_custom(actidx, 'out4d', self.v_dict[lidx-1], array, val, out_q=self.q_dict[lidx-1])
            return obj, cons, array2d
        return 0,0, 0

    def sub_cal_obj_cons(self, lidx, actidx, type_):
        key = get_build_key(lidx, actidx, type_)

        if key in self.subcal_save_dict.keys():
            obj = self.subcal_save_dict[key]['obj'] 
            cons = self.subcal_save_dict[key]['cons'] 
        else:
            if type_=='u':
                obj, cons, array2d = self.sub_cal_obj_cons_u(lidx, actidx)
                self.subcal_save_dict[key] = {'obj' : obj, 'cons' : cons, 'array2d' : array2d}
            elif type_=='v':
                obj, cons = self.sub_cal_obj_cons_v(lidx, actidx)
                self.subcal_save_dict[key] = {'obj' : obj, 'cons' : cons}
        return obj, cons

    def make_subcal_save_dict(self):
        self.logfunc("make subcal start")
        for lidx, nact in enumerate(self.nact_list):
            for actidx in range(nact):
                obj, cons, array2d  = self.sub_cal_obj_cons_u(lidx, actidx)
                key = get_build_key(lidx, actidx, 'u')
                self.subcal_save_dict[key] = {'obj' : obj, 'cons' : cons, 'array2d' : array2d}

                obj, cons = self.sub_cal_obj_cons_v(lidx, actidx)
                key = get_build_key(lidx, actidx, 'v')
                self.subcal_save_dict[key] = {'obj' : obj, 'cons' : cons}

        self.logfunc("make subcal end")

    def save(self, savepath):
        if self.r_dict is not None: 
            r_dict, q_dict, binary_array_list = self.get_solution()
            content = {
                    'r_dict' : r_dict,
                    'q_dict' : q_dict,
                    'binary_array_list' : binary_array_list,
                    'subcal_save_dict' : self.subcal_save_dict
                    }
            write_pkl(content=content, path=savepath) 
        else:
            u_dict, v_dict, q_dict, binary_array_list = self.get_solution()
            content = {
                    'u_dict' : u_dict,
                    'v_dict' : v_dict,
                    'q_dict' : q_dict,
                    'binary_array_list' : binary_array_list,
                    'subcal_save_dict' : self.subcal_save_dict
                    }
            write_pkl(content=content, path=savepath) 

    def load(self, loadpath):
        content = read_pkl(loadpath)
        if 'r_dict' in content.keys():
            r_dict = content['r_dict']
            q_dict = content['q_dict']
            binary_array_list = content['binary_array_list']
            self.subcal_save_dict = content['subcal_save_dict']
            self.init_r_dict_custom(r_dict)
            self.init_q_dict_custom(q_dict)
        else:
            u_dict = content['u_dict']
            v_dict = content['v_dict']
            q_dict = content['q_dict']
            binary_array_list = content['binary_array_list']
            self.subcal_save_dict = content['subcal_save_dict']
            self.init_uv_dict_custom(u_dict, v_dict)
            self.init_q_dict_custom(q_dict)


