import numpy as np

from torch.nn.modules import Module
from .environment_abstract import CubeEnvironment
from utils.pytorch_models import ResnetModel
from sympy.combinatorics import Permutation 


class Cube2x2(CubeEnvironment):
    __name__ = 'cube_2/2/2'
    _moves = {
        'f0': Permutation(23)(2, 19, 21, 8)(3, 17, 20, 10)(4, 6, 7, 5), 
        'f1': Permutation(0, 18, 23, 9)(1, 16, 22, 11)(12, 13, 15, 14), 
        'r0': Permutation(1, 5, 21, 14)(3, 7, 23, 12)(8, 10, 11, 9), 
        'r1': Permutation(23)(0, 4, 20, 15)(2, 6, 22, 13)(16, 17, 19, 18), 
        'd0': Permutation(6, 18, 14, 10)(7, 19, 15, 11)(20, 22, 23, 21), 
        'd1': Permutation(23)(0, 1, 3, 2)(4, 16, 12, 8)(5, 17, 13, 9), 
        '-f0': Permutation(23)(2, 8, 21, 19)(3, 10, 20, 17)(4, 5, 7, 6), 
        '-f1': Permutation(0, 9, 23, 18)(1, 11, 22, 16)(12, 14, 15, 13), 
        '-r0': Permutation(1, 14, 21, 5)(3, 12, 23, 7)(8, 9, 11, 10), 
        '-r1': Permutation(23)(0, 15, 20, 4)(2, 13, 22, 6)(16, 18, 19, 17), 
        '-d0': Permutation(6, 10, 14, 18)(7, 11, 15, 19)(20, 21, 23, 22), 
        '-d1': Permutation(23)(0, 2, 3, 1)(4, 8, 12, 16)(5, 9, 13, 17)}

    def __init__(self, df_info_path, df_puzzles_path):
        super().__init__(df_info_path, df_puzzles_path)
        self.dtype = np.uint8
        self.cube_len = 2
    
    def get_nnet_model(self) -> Module:
        state_dim: int = (self.cube_len ** 2) * 6
        nnet = ResnetModel(state_dim, 6, 8, 5000, 1000, 4, 1, True)
        return nnet
