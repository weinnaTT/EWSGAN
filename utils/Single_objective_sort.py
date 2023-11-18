import numpy as np


def single_sort(target, N, ascending=True):
    '''
    target: minimize
    N: keep
    return the top N
    '''
    b = np.array(target)
    if ascending:
        temp = np.argsort(b)  # 升序排列
    else:
        temp = np.argsort(-b)
    return temp[:N]
