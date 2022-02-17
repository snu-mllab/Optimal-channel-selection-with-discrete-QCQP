import random
import numpy as np
from time_layer import get_scale

def get_idx(group, n):
    idx = 0
    idx2gidx_list = []
    
    for i in range(n):
        visit = False
        for g in group:
            if i in g:
                if i==g[0]:
                    idx2gidx_list.append(idx)
                    idx += 1
                else:
                    idx2gidx_list.append(idx2gidx_list[g[0]])
                visit=True
        if not visit:
            idx2gidx_list.append(idx)
            idx += 1
    
    gidx2idx_list = [0] * (max(idx2gidx_list)+1)
    for idx, gidx in enumerate(idx2gidx_list):
        if gidx2idx_list[gidx]==0:
            gidx2idx_list[gidx]=[idx]
        else: gidx2idx_list[gidx].append(idx)

    return idx2gidx_list, gidx2idx_list
 
def get_random_sample(lists, nsamples, idx2gidx_list, restore=False):
    npos = np.array([len(l) for l in lists])
    numel = 0
    p_list, n_list = [], []
    while len(p_list) < nsamples:
        p = []
        for idx, num in enumerate(npos):
            p.append(lists[idx][random.randrange(0,num)])
        if not restore:
            if p not in p_list:
                p_list.append(p)
    for p in p_list:
        n = []
        for idx in range(len(idx2gidx_list)):
            n.append(p[idx2gidx_list[idx]])
        n_list.append(n)
    return n_list
     
def gen_n_list(actinfo, input_shape_list, min_ratio=0.0, max_ratio=1.0, num_n=100, group=[[1,3,5],[7,9],[11,13],[15,17]], n=19):
    n_list = []
    idx2gidx_list, gidx2idx_list = get_idx(group, n)
    print(idx2gidx_list, gidx2idx_list)
    possible_lists = [0] * len(gidx2idx_list)

    for gidx in range(len(gidx2idx_list)):
        if gidx2idx_list[gidx][0]==n-1:
            possible_lists[gidx] = [1000]
        elif gidx > 0:
            g = gidx2idx_list[gidx]
            nact = actinfo[g[0]]
            inf_h, inf_w = input_shape_list[g[0]]
            scale = get_scale(inf_h)
            possible_lists[gidx] = []
            possible_n = int(nact*min_ratio) + scale
            while possible_n >= min_ratio*nact and possible_n <= max_ratio*nact:
                possible_lists[gidx].append(possible_n)
                possible_n += scale
        else:
            possible_lists[0] = [3]
    n_list = get_random_sample(possible_lists, num_n, idx2gidx_list)
    return n_list


