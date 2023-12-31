import pandas as pd
from pathlib import Path
import argparse
from sympy.combinatorics import Permutation


def get_all_moves(moves_dict):
    moves_dict = {k: Permutation(v) for k,v in moves_dict.items()}
    moves_dict.update({f'-{k}': v**-1 for k, v in moves_dict.items()})
    return moves_dict

def generate_cube_envs(puzzle_info):
    cube_df = puzzle_info[puzzle_info['puzzle_type'].str.startswith('cube')]
    class_template = """
from .environment_abstract import CubeEnvironment
from sympy.combinatorics import Permutation 


class Cube{0}x{0}(CubeEnvironment):
    __name__ = 'cube_{0}/{0}/{0}'
    _moves = {1}

    def __init__(self, df_info_path, df_puzzles_path):
        super().__init__(df_info_path, df_puzzles_path)
        self.cube_len = {0}
"""

    classes = []
    for i, row in cube_df.iterrows():
        puzzle_type = row.puzzle_type
        moves_dict = eval(row.allowed_moves)
        all_moves = get_all_moves(moves_dict)
        size = puzzle_type.split('/')[-1]
        classes.append((size, class_template.format(size, all_moves)))
    
    
    for size, class_def in classes:
        save_path = Path(__file__).resolve().parent.parent.joinpath(f'environments/cube{size}x{size}.py')
        with open(save_path, 'w') as f:
            f.write(class_def)
                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to generate different environments')

    parser.add_argument('--puzzle_info_path', help='Path for dataframe which contains information about different puzzles')
    parser.add_argument('--env_name', help='Type of environment to generate', default='cube')
    args = parser.parse_args()
    
    puzzle_info = pd.read_csv(args.puzzle_info_path)
    if args.env_name == 'cube':
        generate_cube_envs(puzzle_info)
    else:
        raise NotImplementedError
