from .environment_abstract import WreathEnvironment
from sympy.combinatorics import Permutation 


class Wreath21x21(WreathEnvironment):
    __name__ = 'wreath_21/21'
    _moves = {
        'l': Permutation(39)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'r': Permutation(0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 6, 34, 35, 36, 37, 38, 39),
        '-l': Permutation(39)(0, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1),
        '-r': Permutation(0, 39, 38, 37, 36, 35, 34, 6, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21)
    }

    def __init__(self, df_info_path, df_puzzles_path):
        super().__init__(df_info_path, df_puzzles_path)
        self.wreath_size = 21
