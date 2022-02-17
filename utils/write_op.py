import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import pickle

def write_pkl(content, path):
    '''write content on path with path
    Dependency : pickle
    Args:
        content - object to be saved
        path - string
                ends with pkl
    '''
    with open(path, 'wb') as f:
        print("Pickle is written on %s"%path)
        try: pickle.dump(content, f)
        except OverflowError: pickle.dump(content, f, protocol=4)

def create_dir(dirname):
   '''create directory named dirname
   Dependency : os
   Args:
       dirname - string
                 directory named
   '''
   if not os.path.exists(dirname):
       print("Creating %s"%dirname)
       try:
           os.makedirs(dirname)
       except FileExistsError:
           pass
   else:
       print("Already %s exists"%dirname)

def create_muldir(*args):
   for dirname in args: create_dir(dirname)

