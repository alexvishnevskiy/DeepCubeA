from .environment_abstract import WreathEnvironment
from sympy.combinatorics import Permutation 
import numpy as np


class Wreath33x33(WreathEnvironment):
    __name__ = 'wreath_33/33'
    _moves = {
        'l': Permutation(63)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32),
        'r': Permutation(0, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 9, 55, 56, 57, 58, 59, 60, 61, 62, 63),
        '-l': Permutation(63)(0, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1),
        '-r': Permutation(0, 63, 62, 61, 60, 59, 58, 57, 56, 55, 9, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33)
    }

    def __init__(self, df_info_path, df_puzzles_path, num_wildcards = 0):
        super().__init__(df_info_path, df_puzzles_path)
        self.num_wildcards = num_wildcards
        self.wreath_size = 33
