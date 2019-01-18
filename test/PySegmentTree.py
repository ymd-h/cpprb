import numpy as np
import unittest
from ymd_K import SegmentTree

sg = PySegmentTree(16,lambda a,b: a+b)

for i in range(16):
    sg[i] = i * 1.0
    print("{}: {}".format(i,sg[i]))

for i in range(16):
    for j in range(i,16):
        print("[{},{}): {}".format(i,j,sg.reduce(i,j)))
