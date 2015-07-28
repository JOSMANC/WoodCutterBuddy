'''
WoodCutterBuddy
'''
import numpy as np


def woodcutterbuddy(counts, sizes, totalsize=8):
    '''
    /Cutting Stock/
    Application of column generation to solve
    the linear optimization problem
    '''
    pieces_total = np.sum(counts)
    pieces_types = len(sizes)
    diag = np.floor((totalsize)/(sizes).astype(float))
    A = np.zeros((pieces_types, pieces_types))
    np.fill_diagonal(A, diag)
    ones_array = np.ones(pieces_types)
    for _ in xrange(100):
        rhs = counts/np.sum(A, axis=1)
        y = np.linalg.solve(A.T, np.ones(pieces_types))
        ep = knapsack(zip(sizes, y), totalsize)
        if ep is None:
            return A, rhs
        else:
            p = np.linalg.solve(A, ep)
            min_coln = None
            min_colv = (counts/np.sum(A, axis=0)).sum()
            for i in xrange(pieces_types):
                if ep[i] != 0:
                    temp = A.copy()
                    temp[:, [i]] = np.array([ep]).T
                    colv = (counts/np.sum(temp, axis=0)).sum()
                    if (colv < min_colv):
                        min_colv = colv
                        min_coln = i
            if min_coln is not None:
                A[:, [min_coln]] = np.array([ep]).T
            else:
                return A, np.round(rhs)


def knapsack(items, C):
    '''
    /Knapsack/
    Application of a dynamic program to solve
    the optimial weight to cost ratio
    '''
    sack = [(0, [0]*len(items))]*(C+1)
    for i, item in enumerate(items):
        size, value = item
        for c in xrange(int(size), C+1):
            sack_old = sack[c-int(size)]
            sack_trail = sack_old[0] + value
            if sack_trail > sack[c][0]:
                sack[c] = (sack_trail, sack_old[1][:])
                sack[c][1][i] += 1

    if (sum([(sack[C][1][i])*(items[i][0])
            for i in xrange(len(items))])) <= C:
        return np.array(sack[C][1])
    else:
        return None


# if __name__ == "__main__":
#     no = np.array([1, 2, 1])
#     wf = np.array([2, 3, 6])
#     print woodcutterbuddy(no, wf)
