import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import numpy as np

def get_nact_list(array_list):
    '''
    Args:
        array_list - list of 2d array
    '''
    nact_list = list()
    nlist = len(array_list)
    first_array = array_list[0]
    nout, nin = np.shape(first_array)[:2]

    nact_list.append(nin)
    nact_list.append(nout)

    for idx in range(1, nlist):
        array = array_list[idx]
        nout, nin = np.shape(array)[:2]
        assert nact_list[-1] == nin, "Mismatch between out of {} in of {} array list".format(idx-1, idx)
        nact_list.append(nout)
    return nact_list

