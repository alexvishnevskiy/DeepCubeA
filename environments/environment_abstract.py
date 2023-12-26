from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from random import randrange
import pandas as pd
from sympy.combinatorics import Permutation 
import torch.nn as nn


class State(ABC):
    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class Environment(ABC):
    def __init__(self):
        self.dtype = np.float64
        self.fixed_actions: bool = True

    @abstractmethod
    def next_state(self, states: List[State], action: int) -> Tuple[List[State], List[float]]:
        """ Get the next state and transition cost given the current state and action

        @param states: List of states
        @param action: Action to take
        @return: Next states, transition costs
        """
        pass

    @abstractmethod
    def prev_state(self, states: List[State], action: int) -> List[State]:
        """ Get the previous state based on the current state and action

        @param states: List of states
        @param action: Action to take to get the previous state
        @return: Previous states
        """
        pass

    @abstractmethod
    def generate_goal_states(self, num_states: int) -> List[State]:
        """ Generate goal states

        @param num_states: Number of states to generate
        @return: List of states
        """
        pass

    @abstractmethod
    def is_solved(self, states: List[State]) -> np.ndarray:
        """ Returns whether or not state is solved

        @param states: List of states
        @return: Boolean numpy array where the element at index i corresponds to whether or not the
        state at index i is solved
        """
        pass

    @abstractmethod
    def state_to_nnet_input(self, states: List[State]) -> List[np.ndarray]:
        """ State to numpy arrays to be fed to the neural network

        @param states: List of states
        @return: List of numpy arrays. Each index along the first dimension of each array corresponds to the index of
        a state.
        """
        pass

    @abstractmethod
    def get_num_moves(self) -> int:
        """ Used for environments with fixed actions. Corresponds to the numbers of each action

        @return: List of action ints
        """
        pass

    @abstractmethod
    def get_nnet_model(self) -> nn.Module:
        """ Get the neural network model for the environment

        @return: neural network model
        """
        pass

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[State], List[int]]:
        """ Generate training states by starting from the goal and taking actions in reverse.
        If the number of actions are not fixed, then a custom implementation must be used.

        @param num_states: Number of states to generate
        @param backwards_range: Min and max number times to take a move in reverse
        @return: List of states
        """
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states: List[State] = self.generate_goal_states(num_states)

        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        while np.max(num_back_moves < scramble_nums):
            idxs: np.ndarray = np.where((num_back_moves < scramble_nums))[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_to_move = [states[i] for i in idxs]
            states_moved = self.prev_state(states_to_move, move)

            for state_moved_idx, state_moved in enumerate(states_moved):
                states[idxs[state_moved_idx]] = state_moved

            num_back_moves[idxs] = num_back_moves[idxs] + 1

        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        """ Generate all children for the state

        @param states: List of states
        @return: Children of each state, Transition costs for each state
        """
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = []
        for _ in range(len(states)):
            states_exp.append([])

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        move: int
        for move_idx in range(num_env_moves):
            # next state
            states_next_move: List[State]
            tc_move: List[float]
            states_next_move, tc_move = self.next_state(states, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(states_next_move[idx])

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l


class SantaState(State):
    __slots__ = ['state', 'hash']

    def __init__(self, state:np.ndarray):
        if isinstance(state, list):
            state = np.array(state)
        self.state = state
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.state.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)


class SantaEnvironment(Environment):
    def __init__(self, df_info_path, df_puzzles_path):
        super().__init__()
        # initialize moves and goal state
        self._set_goal_state(df_puzzles_path)
        self._init_mappings()
        
    def _set_goal_state(self, df_puzzles_path):
        df_puzzles = pd.read_csv(df_puzzles_path)
        sol_state_str = df_puzzles[df_puzzles['puzzle_type'] == self.__name__].iloc[0]['solution_state']
        sol_state_list = sol_state_str.split(';')
        
        # get mapping color -> int
        self._letter2int = {}
        i = 0
        for color in sol_state_list:
            if color not in self._letter2int:
                self._letter2int[color] = i
                i += 1
        
        self.goal_state = np.array([self._letter2int[color] for color in sol_state_list])
    
    def _init_mappings(self):
        self._int2moves = dict(zip(range(len(self._moves)), self._moves.keys()))
        self._moves2int = {v: k for k, v in self._int2moves.items()}
    
    def get_action(self, action: int) -> Permutation:
        return self._moves[self._int2moves[action]]
        
        
