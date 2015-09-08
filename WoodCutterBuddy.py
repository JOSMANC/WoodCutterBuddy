'''
WoodCutterBuddy
'''
from collections import Counter
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np


class WoodCutterBuddy(object):
    '''
    Class to determine number of standard lumber pieces are required to
    complete a project most efficiently given that the shape of the
    final project materials are know.

    It perform cutting-stock optimization for n wood pieces
    each r inches long.  The class utilizes delayed column generation
    and a revised simplex method with a modified knapsack solver.
    '''
    def __init__(self, plot=True):
        # list of the number of wood size pieces requested
        self.counts = None
        # list of wood sizes request
        self.sizes = None
        # list of wood sizes with a cut error added
        self.error_sizes = None
        # int of total piece of stock lumber
        self.totalsize = None
        # float of how precise should the knapsack algorithm be
        self.precision = None
        # bool indicating whether to plot or not
        self.plot = plot
        # numpy array with the final cut data
        self.final_cuts = None
        # string with the final cut data
        self.final_cuts_string = None

    def cutter(self, counts, sizes, error=0.05,
               precision=20000., totalsize=8., maxsteps=100):
        '''
        /Cutting Stock/
        Application of column generation to solve
        the linear optimization problem
        '''
        self.counts = counts
        self.sizes = sizes
        self.error_sizes = sizes
        self.totalsize = totalsize
        self.precision = precision

        if max(self.error_sizes) > self.totalsize:
            return 'A cut of wood is too larger'

        # total size of all lumber required
        pieces_total = np.sum(self.counts)
        # number of unique pieces which are required
        pieces_types = len(self.sizes)
        # initialize the patters A for cutting the stock
        diag = np.floor((totalsize)/(self.error_sizes).astype(float))
        A = np.zeros((pieces_types, pieces_types))
        np.fill_diagonal(A, diag)
        ones_array = np.ones(pieces_types)
        for _ in xrange(100):
            # rhs for Ax = b
            rhs = self.counts/np.sum(A, axis=1)
            # (sum^m)_i (yi * ai) > 1
            y = np.linalg.solve(A.T, np.ones(pieces_types))
            # entering column to test
            ep = self.knapsack(zip(self.error_sizes, y), self.totalsize)
            # now test if it improves cost function
            min_coln = None
            min_colv = (self.counts/np.sum(A, axis=0)).sum()
            for lc in xrange(pieces_types):
                if ep[lc] != 0:
                    temp = A.copy()
                    temp[:, [lc]] = np.array([ep]).T
                    colv = (self.counts/np.sum(temp, axis=0)).sum()
                    if colv < min_colv:
                        min_colv = colv
                        min_coln = lc
            # if it does update and repeat
            if min_coln is not None:
                A[:, [min_coln]] = np.array([ep]).T
            # if it does not exit
            else:
                return self._display(A, rhs)

    def knapsack(self, items_us, C):
        '''
        /Knapsack/
        Application of a dynamic program to solve
        the optimial weight to cost ratio
        '''
        items = [(int(item[0]*self.precision), item[1])
                 for item in items_us]
        C = int(C*self.precision)
        sack = dict(zip(xrange(C+1), [(0, [0]*len(items))]*(C+1)))
        for i, item in enumerate(items):
            size, value = item
            for c in range(size, C+1):
                sack_old = sack[c-int(size)]
                sack_trial = sack_old[0] + value
                if sack_trial > sack[c][0]:
                    sack[c] = (sack_trial, sack_old[1][:])
                    sack[c][1][i] += 1
        return np.array(sack[C][1])

    def _display(self, A, rhs):
        '''
        /Format the cuts to be displayed/
        Re-arrange the cutting matrix so it can be displayed
        '''
        print A
        #print rhs
        rhs = (np.ceil(rhs)).astype(int)
        A = A.astype(int)
        wood_pieces = []
        maxcuts = A.sum(axis=0).max()
        for i, c in enumerate(rhs):
            for j in xrange(c):
                ctype = []
                for k, t in enumerate(A[:, i]):
                    ctype += [self.sizes[k]]*t
                ctype = filter(lambda a: a != 0, ctype)
                wood_pieces.append(ctype)
        wood_pieces = self._reset_woodpieces(wood_pieces)
        wood_pieces = np.round(wood_pieces, 3)
        order = wood_pieces.sum(axis=1).argsort()
        len_wp = len(wood_pieces)
        best_wp = wood_pieces[order, :]
        for combo in permutations(range(len_wp)):
            temp_wp = self._filter_int_error(wood_pieces[combo, :])
            if len_wp > len(temp_wp):
                best_wp = temp_wp
        wood_pieces = best_wp
        #wood_piecesB = self._filter_int_error(wood_pieces)
        #if len(wood_piecesA) < len(wood_piecesB):
        #    wood_pieces = wood_piecesA
        #else:
        #    wood_pieces = wood_piecesB
        textout = ''
        for i, wp in enumerate(wood_pieces):
            textout += str(i+1) + ':' + str(wp[wp != 0]) + '\n'
        self.final_cuts_array = wood_pieces
        self.final_cuts_string = textout

        #Will not show wood if too many pieces need to be rendered
        if self.plot:
            if len(wood_pieces) <= 6:
                return self.plot_wood(wood_pieces)
        else:
            return self.final_cuts_string

    def _filter_int_error(self, x):
        '''
        /Attempt to fix int errors/
        '''
        check = Counter(dict(zip(self.sizes, self.counts)))
        test = Counter(dict(zip(self.sizes, np.zeros(len(self.sizes)))))
        xnew = []
        waste = []
        for j, twofour in enumerate(x):
            newtwofour = x[0, :]*0
            for i, chunk in enumerate(twofour):
                if test[chunk] != check[chunk]:
                    test[chunk] += 1
                    newtwofour[i] = chunk
            if sum(newtwofour) != 0:
                waste.append(8.-sum(newtwofour))
                xnew.append(newtwofour)
        order = np.argsort(np.array(waste))
        xnew = np.array(xnew)
        return xnew[order, :]

    def _reset_woodpieces(self, wpstart):
        '''
        /Help fill in zeros/
        Make a zero-padded matrix
        '''
        wpieces = 0
        for w in wpstart:
            if len(w) > wpieces:
                wpieces = len(w)

        wp = np.zeros((len(wpstart), wpieces))
        for i, w in enumerate(wpstart):
            l = len(w)
            wp[i, 0:l] = w
        return wp

    def plot_wood(self, wp):
        '''
        /Plot the cut wood/
        Plot the wood and how it should be cut
        '''
        wp = wp.T
        wpieces = wp.shape[1]
        plt.rc('font', family='sans-serif')
        plt.rc('font', serif='Helvetica Neue')
        plt.rc('text', usetex='false')
        plt.rcParams['figure.autolayout'] = True
        labels = []
        for i in range(wpieces):
            l = ''
            l += (''.join([str(w) + "', " for w in wp[:, i] if w != 0]))
            if self.totalsize-wp[:, i].sum() > .1:
                l += '+'+str(self.totalsize-wp[:, i].sum())+"'"
            else:
                l = l[:-2]
            labels.append(l)

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(111)

        ind = np.arange(wpieces)
        bottom = np.zeros(wpieces)
        handles = []
        for c in wp:
            ax1.barh(ind, c, 1, left=bottom, color='#663300',
                     edgecolor='#f1d5b9', linewidth=5)
            bottom += c
        ax1.barh(ind, self.totalsize-wp.sum(axis=0),
                 1, left=bottom, color='#7d280b',
                 edgecolor='#f1d5b9', linewidth=5)
        plt.yticks(ind + 0.5, labels)
        plt.xticks(np.arange(0, 9),
                   [str(i)+"'" for i in np.arange(0, 9)])
        plt.tick_params(axis='x', length=0,
                        labelsize=20, labeltop='off',
                        labelbottom='on')
        plt.tick_params(axis='y', length=0,
                        labelright='on', labelleft='off',
                        labelsize=20)
        ax1.spines['top'].set_linewidth(0)
        ax1.spines['bottom'].set_linewidth(0)
        ax1.spines['left'].set_linewidth(0)
        ax1.spines['right'].set_linewidth(0)
        plt.xlim(0, self.totalsize+.2)
        plt.savefig('woodbuddyschematic.png')
        return self.final_cuts_string

if __name__ == "__main__":
    no = np.array([3, 3, 2, 3])
    wf = np.array([3.5, 4.5, 3.2, 4.])
    wcb = WoodCutterBuddy(plot=True)
    print wcb.cutter(no, wf)
