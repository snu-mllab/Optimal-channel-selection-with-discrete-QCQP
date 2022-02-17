import numpy as np

from utils.pytorch_img_op import tensor2numpy 
from utils.resnet_dict import get_dict
from utils.pytorch_op import get_num_param, get_zero_param, update_target, init_module_weight_2_one
from utils.prun_op import prun_skip_2D_single, prun_skip_4D
import torch

class MaskManager_skip:
    def __init__(self):
        self.param_dict = dict()
        self.key_array = list()
        self.u_dict = dict()
        self.v_dict = dict()
        self.q_dict = dict()
        self.arch = None
    
    def init(self):
        self.model_dict, self.val_dict = get_dict(self.arch)
        self.s_list = self.model_dict['s_list']
        self.t_list = self.model_dict['t_list']

    def weight2mask(self):
        self.masking()
        self.initialize_mask()
        self.weight2uvq()
        self.uvq2mask()

    def weight2uvq(self):
        self.u_dict[0] = np.ones(3)
        for idx, key in enumerate(self.key_array):
            if 'fc' in key:
                origin = tensor2numpy(self.param_dict[key]['origin'].weight.data)
                outf, inf = origin.shape
                inact = (np.max(np.abs(origin), axis=0)>0)*1.0
                outact = (np.max(np.abs(origin), axis=1)>0)*1.0
                self.u_dict[idx+1] = outact
                self.v_dict[idx] = inact
            elif 'conv' in key:
                origin = tensor2numpy(self.param_dict[key]['origin'].weight.data)
                outf, inf, h, w = origin.shape
                infilact = (np.max(np.abs(origin), axis=(0,2,3))>0)*1.0 # [InFilter]
                outfilact = (np.max(np.abs(origin), axis=(1,2,3))>0)*1.0 # [OutFilter]
                ptfilact = (np.max(np.abs(origin), axis=(1))>0)*1.0
                self.u_dict[idx+1] = outfilact
                self.v_dict[idx] = infilact
                self.q_dict[idx] = ptfilact
        self.v_dict[len(self.key_array)] = self.u_dict[len(self.key_array)]

    def uvq2mask(self):
        self.initialize_mask()
        for idx, key in enumerate(self.key_array):
            if 'fc' in key:
                mask = self.param_dict[key]['mask']
                mask_np = tensor2numpy(mask.weight.data)
                binary = np.zeros_like(mask_np)
                nout, nin = np.shape(binary)
                for outidx in range(nout):
                    for inidx in range(nin):
                        if self.u_dict[idx+1][outidx]==1 and self.v_dict[idx][inidx]==1:
                            binary[outidx][inidx]=1
                update_target(mask.weight,binary)
            elif 'conv' in key:
                mask = self.param_dict[key]['mask']
                mask_np = tensor2numpy(mask.weight.data)
                binary = np.zeros_like(mask_np)
                nout, nin, height, width = np.shape(binary)
                for outidx in range(nout):
                    for inidx in range(nin):
                        for hidx in range(height):
                            for widx in range(width):
                                if self.u_dict[idx+1][outidx]==1 and self.v_dict[idx][inidx]==1 and self.q_dict[idx][outidx][hidx][widx]==1:
                                    binary[outidx][inidx][hidx][widx]=1
                update_target(mask.weight,binary)
    
    def uv2mask(self):
        self.initialize_mask()
        for idx, key in enumerate(self.key_array):
            if 'fc' in key:
                mask = self.param_dict[key]['mask']
                mask_np = tensor2numpy(mask.weight.data)
                binary = np.zeros_like(mask_np)
                nout, nin = np.shape(binary)
                for outidx in range(nout):
                    for inidx in range(nin):
                        if self.u_dict[idx+1][outidx]==1 and self.v_dict[idx][inidx]==1:
                            binary[outidx][inidx]=1
                update_target(mask.weight,binary)
            elif 'conv' in key:
                mask = self.param_dict[key]['mask']
                mask_np = tensor2numpy(mask.weight.data)
                binary = np.zeros_like(mask_np)
                nout, nin, height, width = np.shape(binary)
                for outidx in range(nout):
                    for inidx in range(nin):
                        for hidx in range(height):
                            for widx in range(width):
                                if self.u_dict[idx+1][outidx]==1 and self.v_dict[idx][inidx]==1:
                                    binary[outidx][inidx][hidx][widx]=1
                update_target(mask.weight,binary)

    def add(self, key, origin, mask, input_size, output_size):
        self.param_dict[key] = { 'origin' : origin, 'mask' :  mask, 'input' : np.ones(input_size), 'output' : np.ones(output_size)}
        self.key_array.append(key)

    def initialize_mask(self, select=None):
        for key in self.key_array:
            if select is None or select in key:
                init_module_weight_2_one(self.param_dict[key]['mask'])

    def masking(self):
        for key in self.key_array:
            origin = self.param_dict[key]['origin']
            mask = self.param_dict[key]['mask']
            origin.weight.data = torch.mul(origin.weight, mask.weight)
    
    def get_weights_and_flops_and_arrays(self, start_idx, end_idx, dim = '2D'):
        if dim=='1D':
            weight_list=list()
            array_list=list()
            for kidx in range(start_idx, end_idx):
                key = self.key_array[kidx]
                origin = self.param_dict[key]['origin']
                input_act = self.param_dict[key]['input']
                output_act = self.param_dict[key]['output']
                weight = tensor2numpy(origin.weight.data)
                weight_list.append(weight)
            for weight in weight_list:
                v = np.sum(np.abs(weight),axis=(1,2,3))
                v = v/np.sqrt(np.sum(v*v))
                nout = v.shape[0]
                for outidx in range(nout):
                    if v[outidx] != v[outidx]:
                        v[outidx] = 0
                array_list.append(v)
            return array_list    
        elif dim == '2D':
            flops_const_list = list()
            weight_list = list()
            array_list = list()
            for kidx in range(start_idx, end_idx):
                key = self.key_array[kidx]
                origin = self.param_dict[key]['origin']
                input_act = self.param_dict[key]['input']
                output_act = self.param_dict[key]['output']
                weight = tensor2numpy(origin.weight.data)
                k = weight.shape[2]
                h = output_act.shape[1]
                w = output_act.shape[2]
                flops = k*k*h*w
                weight_list.append(weight)
                flops_const_list.append(flops)
            for weight in weight_list:
                v = np.sum(np.abs(weight),axis=(2,3))
                v = v/np.sqrt(np.sum(v*v))
                nout, nin = v.shape
                for outidx in range(nout):
                    for inidx in range(nin):
                        if v[outidx][inidx] != v[outidx][inidx]:
                            v[outidx][inidx] = 0
                array_list.append(v)
            
            return weight_list, flops_const_list, array_list
        elif dim == '4D':
            flops_const_list = list()  
            weight_list = list() 
            array_list = list()
            for kidx in range(start_idx, end_idx):
                key = self.key_array[kidx]
                origin = self.param_dict[key]['origin']
                input_act = self.param_dict[key]['input']
                output_act = self.param_dict[key]['output']
                weight = tensor2numpy(origin.weight.data)
                k = weight.shape[2]
                h = output_act.shape[1]
                w = output_act.shape[2]
                weight_list.append(weight)
                flops_const_list.append(h*w)
            for weight in weight_list:
                v = np.abs(weight)
                v = v/np.sqrt(np.sum(v*v))
                nout, nin, h, w = v.shape
                for outidx in range(nout):
                    for inidx in range(nin):
                        for hidx in range(h):
                            for widx in range(w):
                                if v[outidx][inidx][hidx][widx] != v[outidx][inidx][hidx][widx]:
                                    v[outidx][inidx][hidx][widx] = 0
                array_list.append(v)
            return weight_list, flops_const_list, array_list
        
   
    def get_flops_constraint(self, start_idx, end_idx, flops_c):
        max_flops = self.get_max_flop() # max flops (don't consider downsampling)
        range_flops = self.get_max_flop_with_range(start_idx, end_idx) # max flops in conv (except downsampling)
        flops_constraint = flops_c * max_flops - (max_flops - range_flops) # flops constraint in conv assuming fc part is maximal
        if flops_constraint < 0 : print("flops_c is too small")
        print("max_flops : {}, range_flops : {}, flops_c : {}, flops_constraint : {}".format(max_flops, range_flops, flops_c, flops_constraint))
        return flops_constraint

    def get_mem_constraint(self, start_idx, end_idx, mem_c, batch_size=1):
        max_mem = self.get_max_mem(batch_size=batch_size) # max mem (don't consider downsampling)
        range_mem = self.get_max_mem_with_range(start_idx, end_idx, batch_size=batch_size) # max mem in conv (except downsampling)
        mem_constraint = mem_c * max_mem - (max_mem - range_mem) # mem constraint in conv assuming fc part is maximal
        if mem_constraint < 0 : print("mem_c is too small")
        print("batch_size : {}, max_mem : {}, range_mem : {}, mem_c : {}, mem_constraint : {}".format(batch_size,max_mem, range_mem, mem_c, mem_constraint))
        return mem_constraint

    def convuv2mask(self, u_dict, v_dict):
        l = len(u_dict.keys())
        if v_dict[l-1].shape[0] == 64:
            u_dict[l] = np.ones(10)
            v_dict[l] = np.ones(10)
        else:
            u_dict[l] = np.ones(1000)
            v_dict[l] = np.ones(1000)
        self.u_dict = u_dict
        self.v_dict = v_dict
        self.uv2mask()

    def convuvq2mask(self, u_dict, v_dict, q_dict):
        l = len(u_dict.keys())
        if v_dict[l-1].shape[0] == 64:
            u_dict[l] = np.ones(10)
            v_dict[l] = np.ones(10)
        else:
            u_dict[l] = np.ones(1000)
            v_dict[l] = np.ones(1000)
        self.u_dict = u_dict
        self.v_dict = v_dict
        self.q_dict = q_dict
        self.uvq2mask()

    def prun_flops_with_weight(self,  
            start_idx, 
            end_idx, 
            flops_c, 
            use_cplex=False,
            timelimit=None,
            path=None,
            debug=False,
            ):
        weight_list, flops_const_list, array_list = self.get_weights_and_flops_and_arrays(start_idx,end_idx,dim='2D')
        flops_constraint = self.get_flops_constraint(start_idx, end_idx, flops_c)
        u_dict,v_dict = prun_skip_2D_single(
                array_list=array_list,
                s_list=self.s_list,
                t_list=self.t_list,
                quadratic_val_list=flops_const_list,
                max_constraint=flops_constraint,
                use_cplex=use_cplex,
                timelimit=timelimit,
                path=path,
                )
        self.convuv2mask(u_dict,v_dict)
        self.masking() 

    def prun_flops_with_weight_4D(self,  
            start_idx, 
            end_idx, 
            flops_c, 
            gamma,
            use_cplex=False,
            timelimit=None,
            path=None,
            debug=False,
            ):
        weight_list, flops_const_list, array_list = self.get_weights_and_flops_and_arrays(start_idx,end_idx,dim='4D')
        flops_constraint = self.get_flops_constraint(start_idx, end_idx, flops_c)
        u_dict,v_dict,q_dict = prun_skip_4D(
                array_list=array_list,
                s_list=self.s_list,
                t_list=self.t_list,
                quadratic_val_list=flops_const_list,
                max_constraint=flops_constraint,
                gamma=gamma,
                use_cplex=use_cplex,
                timelimit=timelimit,
                path=path,
                )
        self.convuvq2mask(u_dict,v_dict,q_dict)
        self.masking() 

    def prun_mem_with_weight(self,
            start_idx,
            end_idx,
            mem_c,
            use_cplex=False,
            timelimit=None,
            path=None,
            debug=False,
            ):
        weight_list, flops_const_list, array_list = self.get_weights_and_flops_and_arrays(start_idx,end_idx,dim='2D')
        mem_const_list = self.val_dict['quadratic_val_list_mem_2D'][:-1]
        mem_constraint = self.get_mem_constraint(start_idx, end_idx, mem_c, batch_size=0)
        if debug : 
            print("mem_const_list : {}".format(mem_const_list))
            print("u_linear_val_list_mem : {}".format(u_linear_val_list_mem))
            print("v_linear_val_list_mem : {}".format(v_linear_val_list_mem))
        u_dict, v_dict = prun_skip_2D_single(
                array_list=array_list,
                s_list=self.s_list,
                t_list=self.t_list,
                quadratic_val_list=mem_const_list,
                max_constraint=mem_constraint,
                use_cplex=use_cplex,
                timelimit=timelimit,
                path=path,
                )
        self.convuv2mask(u_dict,v_dict)
        self.masking()

    def prun_mem_with_weight_4D(self,
            start_idx,
            end_idx,
            mem_c,
            gamma,
            use_cplex=False,
            timelimit=None,
            path=None,
            debug=False,
            ):
        weight_list, flops_const_list, array_list = self.get_weights_and_flops_and_arrays(start_idx,end_idx,dim='4D')
        mem_const_list = self.val_dict['quadratic_val_list_mem_4D'][:-1]
        mem_constraint = self.get_mem_constraint(start_idx, end_idx, mem_c, batch_size=0)
        if debug : 
            print("mem_const_list : {}".format(mem_const_list))
            print("batch_size : {}".format(batch_size))
        u_dict, v_dict, q_dict = prun_skip_4D(
                array_list=array_list,
                s_list=self.s_list,
                t_list=self.t_list,
                quadratic_val_list=mem_const_list,
                max_constraint=mem_constraint,
                gamma=gamma,
                use_cplex=use_cplex,
                timelimit=timelimit,
                path=path,
                )
        self.convuvq2mask(u_dict,v_dict,q_dict)
        self.masking()

    def get_totalcount(self):
        total_count = 0
        for key in self.key_array:
            origin = self.param_dict[key]['mask']
            total_count += get_num_param([origin.weight])
        return total_count

    def get_zerocount(self):
        zero_count = 0
        for key in self.key_array:
            origin = self.param_dict[key]['mask']
            zero_count += get_zero_param([origin.weight])
        return zero_count

    def get_max_flop_with_range(self, sidx, eidx, debug=False):
        self.masking()
        flop = 0 
        for kidx in range(sidx, eidx):
            key = self.key_array[kidx]
            origin = self.param_dict[key]['origin']
            input_act = self.param_dict[key]['input']
            output_act = self.param_dict[key]['output']

            if 'fc' in key:
                nin = input_act.shape[0]
                nout = output_act.shape[0]
                flop += nin * nout
                if debug:
                    print("fc_flops added : {}".format(nin*nout))
            elif 'conv' in key:
                outf, inf, h, w = tensor2numpy(self.param_dict[key]['origin'].weight.data).shape
                nin =  input_act.shape[0] 
                nout = output_act.shape[0]
                flop += h*w*nin*nout*output_act.shape[1]*output_act.shape[2]
                if debug:
                    print("conv_flops added :{}".format(h*w*nin*nout*output_act.shape[1]*output_act.shape[2]))
            if debug:
                print("origin_flops : {}".format(flop))
        return flop 
    
    def get_max_flop(self):
        self.masking()
        flop = 0 
        for key in self.key_array:
            origin = self.param_dict[key]['origin']
            input_act = self.param_dict[key]['input']
            output_act = self.param_dict[key]['output']

            if 'fc' in key:
                nin = input_act.shape[0]
                nout = output_act.shape[0]
                flop += nin * nout
            elif 'conv' in key:
                outf, inf, h, w = tensor2numpy(self.param_dict[key]['origin'].weight.data).shape
                nin =  input_act.shape[0]
                nout = output_act.shape[0]
                flop += h*w*nin*nout*output_act.shape[1]*output_act.shape[2]
        return flop 
    
    def get_max_mem_with_range(self,sidx,eidx,debug=False, batch_size=1):
        self.masking()
        quadratic_val_list_mem_4D = self.val_dict['quadratic_val_list_mem_4D']
        max_mem = 0
        # quadratic
        for i in range(sidx,eidx):
            key = self.key_array[i]
            mask = tensor2numpy(self.param_dict[key]['mask'].weight.data)
            num_tot = np.sum(np.ones_like(mask))
            max_mem += quadratic_val_list_mem_4D[i] * num_tot
        return max_mem

    def get_max_mem(self,debug=False, batch_size=1):
        self.masking()
        quadratic_val_list_mem_4D = self.val_dict['quadratic_val_list_mem_4D']
        max_mem = 0
        L = len(self.key_array)+1
        # quadratic
        for i in range(L-1):
            key = self.key_array[i]
            mask = tensor2numpy(self.param_dict[key]['mask'].weight.data)
            num_tot = np.sum(np.ones_like(mask))
            max_mem += quadratic_val_list_mem_4D[i] * num_tot
        return max_mem

    def evaluate_from_uvq_and_mask(self, debug=False, printf=print, batch_size=1):
        
        # get info
        quadratic_val_list_flops_2D = self.val_dict['quadratic_val_list_flops_2D']
        quadratic_val_list_flops_4D = self.val_dict['quadratic_val_list_flops_4D']
        const_flops = self.val_dict['const_flops']
        
        total_flops = 0
        max_flops = 0
        # quadratic
        for i, key in enumerate(self.key_array):
            mask = tensor2numpy(self.param_dict[key]['mask'].weight.data)
            origin = tensor2numpy(self.param_dict[key]['origin'].weight.data)
            num_count = get_num_param([self.param_dict[key]['origin'].weight]) - get_zero_param([self.param_dict[key]['origin'].weight])
            num_mask = np.sum(mask)
            num_tot = np.sum(np.ones_like(mask))
            max_flops += quadratic_val_list_flops_4D[i] * num_tot
            total_flops += quadratic_val_list_flops_4D[i] * num_mask
        pruning_ratio = self.get_zerocount()/self.get_totalcount()
        content = dict()
        content['max_flops'] = max_flops
        content['total_flops'] = total_flops
        content['pruning_ratio'] = pruning_ratio
        content['flops_ratio'] = total_flops / max_flops
        printf("max_flops : {}, total_flops : {}, {}% flops used".format(max_flops,total_flops,total_flops/max_flops*100))
        printf("pruning_ratio : {}%".format(pruning_ratio*100))
        return content

    def evaluate_from_weight(self, debug=False,printf=print,batch_size=1):
        self.masking()
        self.weight2mask()
        return self.evaluate_from_uvq_and_mask(printf=printf,batch_size=batch_size)

