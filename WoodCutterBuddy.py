'''
WoodCutterBuddy
'''
import numpy as np
import matplotlib.pyplot as plt


class WoodCutterBuddy(object):

    def __init__(self, plot=True):
        self.counts = None
        self.sizes = None
        self.error_sizes = None
        self.totalsize = None
        self.precision = None
        self.plot = True
        self.final_cuts = None
        self.final_cuts_string = None

    def cutter(self, counts, sizes, error=0.05,
                precision=1000, totalsize=8, maxsteps=100):
        '''
        /Cutting Stock/
        Application of column generation to solve
        the linear optimization problem
        '''
        self.counts = counts
        self.sizes = sizes
        self.error_sizes = sizes + error
        self.totalsize = totalsize
        self.precision = precision
        self.maxsteps = maxsteps

        if max(self.error_sizes) > self.totalsize:
            return 'A cut of wood is too larger'

        pieces_total = np.sum(self.counts)
        pieces_types = len(self.sizes)
        diag = np.floor((totalsize)/(self.error_sizes).astype(float))
        A = np.zeros((pieces_types, pieces_types))
        np.fill_diagonal(A, diag)
        ones_array = np.ones(pieces_types)
        for _ in xrange(maxsteps):
            rhs = self.counts/np.sum(A, axis=1)
            y = np.linalg.solve(A.T, np.ones(pieces_types))
            ep = self.knapsack(zip(self.error_sizes, y), self.totalsize)
            p = np.linalg.solve(A, ep)
            min_coln = None
            min_colv = (self.counts/np.sum(A, axis=0)).sum()
            for lc in xrange(pieces_types):
                if ep[lc] != 0:
                    temp = A.copy()
                    temp[:, [lc]] = np.array([ep]).T
                    colv = (self.counts/np.sum(temp, axis=0)).sum()
                    if (colv < min_colv):
                        min_colv = colv
                        min_coln = lc
            if min_coln is not None:
                A[:, [min_coln]] = np.array([ep]).T
            else:
                return self._display(A, rhs)

    def knapsack(self, items_us, C):
        '''
        /Knapsack/
        Application of a dynamic program to solve
        the optimial weight to cost ratio
        '''
        items = [(int(item[0]*self.precision), item[1]) for item in items_us]
        C = int(C*self.precision)
        sack = [(0, [0]*len(items))]*(C+1)
        for i, item in enumerate(items):
            size, value = item
            for c in range(size, C+1):
                sack_old = sack[c-int(size)]
                sack_trial = sack_old[0] + value
                if (sack_trial > sack[c][0]):
                    sack[c] = (sack_trial, sack_old[1][:])
                    sack[c][1][i] += 1
        return np.array(sack[C][1])
        return np.array(sack[C][1])

    def _display(self, A, rhs):
        '''
        /Knapsack/
        Application of a dynamic program to solve
        the optimial weight to cost ratio
        '''
        rhs = (np.ceil(rhs)).astype(int)
        A = A.astype(int)
        wood_pieces = []
        maxcuts = A.sum(axis=0).max()
        for i, c in enumerate(rhs):
            for j in xrange(c):
                ctype = []
                for k, t in enumerate(A[:, i]):
                    ctype += [self.sizes[k]]*t
                    #for _ in xrange(maxcuts-len(ctype)):
                        #ctype.append(0.)
                ctype = filter(lambda a: a != 0, ctype)
                wood_pieces.append(ctype)
        wood_pieces = self._reset_woodpieces(wood_pieces)
        wood_pieces = np.round(wood_pieces, 3)
        textout = ''
        for i, wp in enumerate(wood_pieces):
            textout += str(i+1)+':'+str(wp[wp!=0])+'\n'
        order = wood_pieces.sum(axis=1).argsort()
        wood_pieces = wood_pieces[order]
        self.final_cuts_array = wood_pieces
        self.final_cuts_string = textout
        if self.plot:
            if len(wood_pieces) <= 6:
                self.plot_wood(wood_pieces)

    def _reset_woodpieces(self, wpstart):

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
        wp = wp.T
        wpieces = wp.shape[1]
        plt.rc('font', family='sans-serif')
        plt.rc('font', serif='Helvetica Neue')
        plt.rc('text', usetex='false')
        plt.rcParams['figure.autolayout'] =  True
        labels = []
        for i in range(wpieces):
            l = ''
            l += (''.join([str(w)+"', " for w in wp[:, i] if w!=0]))
            if (self.totalsize-wp[:, i].sum()) > .1:
                l += '+'+str(self.totalsize-wp[:, i].sum())+"'"
            else:
                l = l[:-2]
            labels.append(l)

        fig = plt.figure(figsize=(15,4))
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
        plt.yticks(ind+0.5, labels)
        plt.xticks(np.arange(0,9),
                   [str(i)+"'" for i in np.arange(0,9)])
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
        plt.xlim(0, self.totalsize)
        plt.show()


# if __name__ == "__main__":
#     no = np.array([1,    1,   2,  1])
#     wf = np.array([4.5, 3.3, 1.1, 2.])
#     wcb = WoodCutterBuddy(plot=True)
#     print wcb.cutter(no, wf)
