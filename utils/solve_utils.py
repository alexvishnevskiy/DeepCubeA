from typing import List, Tuple, Optional
from search_methods.astar import bwas_python, bwas_cpp
from environments.environment_abstract import SantaEnvironment, SantaState
from utils.env_utils import get_environment
import os


class SantaSolver:
    def __init__(
        self, 
        env_name: str,
        model_dir: str, 
        nnet_batch_size: int = 10000, 
        weight: float = 0.8,
        batch_size: int = 10000,
        verbose: bool = True,
        df_info_path: str = '/mnt/hdd/santa/data/puzzle_info.csv',
        df_puzzles_path: str = '/mnt/hdd/santa/data/puzzles.csv'
    ):
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        
        self.env = get_environment(env_name, df_info_path = df_info_path, df_puzzles_path = df_puzzles_path)
        self.model_dir = model_dir
        self.nnet_batch_size = nnet_batch_size
        self.weight = weight
        self.batch_size = batch_size
        self.verbose = verbose
        
    def __call__(self, state: str, num_wildcards: int):
        santa_state = self.env.strState2SantaState(state)
        self.env.num_wildcards = num_wildcards
        moves, _, _, _ = bwas_python(self, self.env, [santa_state])
        solution = '.'.join([self.env._int2moves[move] for move in moves[0]])
        return solution
