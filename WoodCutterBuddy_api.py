'''
WoodCutterBuddy Executor
'''
from WoodCutterBuddy import WoodCutterBuddy
import sys
import numpy as np
if __name__ == "__main__":
    print 'Let us get cutting!'
    print '--------------------------------------'
    print 'sizes of wood needed e.g. 4. 2.0 2.0'
    sizestxt = raw_input('wood sizes: ')
    sizes = sizestxt.split()
    try:
        sizes = [float(s) for s in sizes]
    except:
        sys.exit("---input error---")
    print ''
    print 'count of wood needed e.g. 1 2 3'
    countstxt = raw_input('cout of sizes: ')
    counts = countstxt.split()
    try:
        counts = [int(c) for c in counts]
    except:
        sys.exit("---input error---")
    print ''
    plottxt = raw_input('Save schematic? (Y/N): ')
    print ''
    if len(counts) != len(sizes):
        sys.exit("---input error---")
    print '--------------------------------------'
    wcb = WoodCutterBuddy(plot=(plottxt=='Y'))
    result = wcb.cutter(np.array(counts), np.array(sizes))
    print 'wood cutting pattern is:'
    print result
    print 'Go cut some wood!'

