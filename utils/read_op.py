import pickle

def read_pkl(path, encoding='ASCII'):
    '''read path(pkl) and return files
    Dependency : pickle
    Args:
        path - string
               ends with pkl
    Return:
        pickle content
    '''
    print("Pickle is read from %s"%path)
    with open(path, 'rb') as f: return pickle.load(f, encoding=encoding)

