from typing import List, Tuple
from numpy import ndarray
import numpy as np

from torch.nn.modules import Module
from .environment_abstract import SantaEnvironment, SantaState
from utils.pytorch_models import ResnetModel
from sympy.combinatorics import Permutation 



class Cube2x2State(SantaState):
    pass


class Cube2x2(SantaEnvironment):
    __name__ = 'cube_2/2/2'
    _moves = {'f0': Permutation(23)(2, 19, 21, 8)(3, 17, 20, 10)(4, 6, 7, 5), 'f1': Permutation(0, 18, 23, 9)(1, 16, 22, 11)(12, 13, 15, 14), 'r0': Permutation(1, 5, 21, 14)(3, 7, 23, 12)(8, 10, 11, 9), 'r1': Permutation(23)(0, 4, 20, 15)(2, 6, 22, 13)(16, 17, 19, 18), 'd0': Permutation(6, 18, 14, 10)(7, 19, 15, 11)(20, 22, 23, 21), 'd1': Permutation(23)(0, 1, 3, 2)(4, 16, 12, 8)(5, 17, 13, 9), '-f0': Permutation(23)(2, 8, 21, 19)(3, 10, 20, 17)(4, 5, 7, 6), '-f1': Permutation(0, 9, 23, 18)(1, 11, 22, 16)(12, 14, 15, 13), '-r0': Permutation(1, 14, 21, 5)(3, 12, 23, 7)(8, 9, 11, 10), '-r1': Permutation(23)(0, 15, 20, 4)(2, 13, 22, 6)(16, 18, 19, 17), '-d0': Permutation(6, 10, 14, 18)(7, 11, 15, 19)(20, 21, 23, 22), '-d1': Permutation(23)(0, 2, 3, 1)(4, 8, 12, 16)(5, 9, 13, 17)}

    def __init__(self, df_info_path, df_puzzles_path):
        super().__init__(df_info_path, df_puzzles_path)
        self.dtype = np.uint8
        self.cube_len = 2
    
    def generate_goal_states(self, num_states: int, np_format: bool = False):
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_state.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[Cube2x2State] = [Cube2x2State(self.goal_state.copy()) for _ in range(num_states)]

        return solved_states
    
    def get_nnet_model(self) -> Module:
        state_dim: int = (self.cube_len ** 2) * 6
        nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True)
        return nnet
    
    def get_num_moves(self) -> int:
        return len(self._moves)
    
    def is_solved(self, states) -> ndarray:
        states_np = np.stack([state.state for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.goal_state, 0))
        return np.all(is_equal, axis=1)
    
    def next_state(self, states: List[Cube2x2State], action: int):
        action = self.get_action(action)
        states_next = [Cube2x2State(action(state.state)) for state in states]
        transition_costs = [1.0 for _ in range(len(states))]
        return states_next, transition_costs
    
    def prev_state(self, states: List[Cube2x2State], action: int):
        str_move = self._int2moves[action] #convert int action to str action
        rev_action = self._moves2int[f'-{str_move}' if str_move[0] != '-' else str_move[1:]] #get reverse action
        return self.next_state(states, rev_action)[0]
    
    def state_to_nnet_input(self, states) -> List[ndarray]:
        states_np = np.stack([state.state for state in states], axis=0)

        representation_np = states_np / (self.cube_len ** 2)
        representation_np = representation_np.astype(self.dtype)

        representation = [representation_np]
        return representation

# a = Cube2x2('/mnt/hdd/santa/data/puzzle_info.csv', '/mnt/hdd/santa/data/puzzles.csv')
# print(a._moves)