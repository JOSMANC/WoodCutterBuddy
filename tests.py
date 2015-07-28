'''
Tests for WoodCutterBuddy
'''
import numpy as np
from WoodCutterBuddy import woodcutterbuddy, knapsack


def test_knapsack():

    items = [(7, 2.9), (30, 1)]
    C = 21
    test1 = np.array_equal([3, 0], np.round(knapsack(items, C)))
    items = [(6, 2.9), (3, 1), (5, 2.)]
    C = 21
    test2 = np.array_equal([3, 1, 0], np.round(knapsack(items, C)))
    if ((test1+test2) == 2):
        print 'knapsack tests passed'


def test_woodcutterbuddy():
    supplies = (np.array([1, 2, 1]), np.array([2, 3, 6]))
    cuts, cut_counts = woodcutterbuddy(supplies[0], supplies[1])
    test1a = np.array_equal(np.array([[4, 1, 1],
                                      [0, 2, 0],
                                      [0, 0, 1]]), cuts)
    test1b = np.array_equal(np.array([0, 1, 1]), cut_counts)
    if ((test1a+test1b) == 2):
        print 'woodcutterbuddy test passed'


if __name__ == "__main__":
    test_knapsack()
    test_woodcutterbuddy()
