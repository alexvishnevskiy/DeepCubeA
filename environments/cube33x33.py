
# This file was generated using utils/env_generation.py
from .environment_abstract import CubeEnvironment
from sympy.combinatorics import Permutation 


class Cube33x33(CubeEnvironment):
    __name__ = 'cube_33/33/33'
    # _moves = should be permutation matrix

    def __init__(self, df_info_path, df_puzzles_path):
        super().__init__(df_info_path, df_puzzles_path)
        self.cube_len = 33
