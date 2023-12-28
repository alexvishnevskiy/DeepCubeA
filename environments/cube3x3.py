import numpy as np

from torch.nn.modules import Module
from .environment_abstract import CubeEnvironment
from utils.pytorch_models import ResnetModel
from sympy.combinatorics import Permutation 



class Cube3x3(CubeEnvironment):
    __name__ = 'cube_3/3/3'
    _moves = {
        'f0': Permutation(53)(6, 44, 47, 18)(7, 41, 46, 21)(8, 38, 45, 24)(9, 15, 17, 11)(10, 12, 16, 14), 
        'f1': Permutation(53)(3, 43, 50, 19)(4, 40, 49, 22)(5, 37, 48, 25), 
        'f2': Permutation(0, 42, 53, 20)(1, 39, 52, 23)(2, 36, 51, 26)(27, 29, 35, 33)(28, 32, 34, 30),
        'r0': Permutation(2, 11, 47, 33)(5, 14, 50, 30)(8, 17, 53, 27)(18, 24, 26, 20)(19, 21, 25, 23),
        'r1': Permutation(53)(1, 10, 46, 34)(4, 13, 49, 31)(7, 16, 52, 28),
        'r2': Permutation(53)(0, 9, 45, 35)(3, 12, 48, 32)(6, 15, 51, 29)(36, 38, 44, 42)(37, 41, 43, 39),
        'd0': Permutation(15, 42, 33, 24)(16, 43, 34, 25)(17, 44, 35, 26)(45, 51, 53, 47)(46, 48, 52, 50),
        'd1': Permutation(53)(12, 39, 30, 21)(13, 40, 31, 22)(14, 41, 32, 23),
        'd2': Permutation(53)(0, 2, 8, 6)(1, 5, 7, 3)(9, 36, 27, 18)(10, 37, 28, 19)(11, 38, 29, 20),
        '-f0': Permutation(53)(6, 18, 47, 44)(7, 21, 46, 41)(8, 24, 45, 38)(9, 11, 17, 15)(10, 14, 16, 12),
        '-f1': Permutation(53)(3, 19, 50, 43)(4, 22, 49, 40)(5, 25, 48, 37),
        '-f2': Permutation(0, 20, 53, 42)(1, 23, 52, 39)(2, 26, 51, 36)(27, 33, 35, 29)(28, 30, 34, 32),
        '-r0': Permutation(2, 33, 47, 11)(5, 30, 50, 14)(8, 27, 53, 17)(18, 20, 26, 24)(19, 23, 25, 21),
        '-r1': Permutation(53)(1, 34, 46, 10)(4, 31, 49, 13)(7, 28, 52, 16),
        '-r2': Permutation(53)(0, 35, 45, 9)(3, 32, 48, 12)(6, 29, 51, 15)(36, 42, 44, 38)(37, 39, 43, 41),
        '-d0': Permutation(15, 24, 33, 42)(16, 25, 34, 43)(17, 26, 35, 44)(45, 47, 53, 51)(46, 50, 52, 48),
        '-d1': Permutation(53)(12, 21, 30, 39)(13, 22, 31, 40)(14, 23, 32, 41),
        '-d2': Permutation(53)(0, 6, 8, 2)(1, 3, 7, 5)(9, 18, 27, 36)(10, 19, 28, 37)(11, 20, 29, 38)
    }

    def __init__(self, df_info_path, df_puzzles_path):
        super().__init__(df_info_path, df_puzzles_path)
        self.dtype = np.uint8
        self.cube_len = 3
    
    def get_nnet_model(self) -> Module:
        state_dim: int = (self.cube_len ** 2) * 6
        nnet = ResnetModel(state_dim, 6, 8, 5000, 1000, 4, 1, True)
        return nnet
