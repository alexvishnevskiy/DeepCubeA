import re
import math
from environments.environment_abstract import Environment


def get_environment(env_name: str, **kwargs) -> Environment:
    env_name = env_name.lower()
    puzzle_n_regex = re.search("puzzle(\d+)", env_name)
    env: Environment

    if env_name == 'cube3':
        from environments.cube3 import Cube3
        env = Cube3()
    elif puzzle_n_regex is not None:
        from environments.n_puzzle import NPuzzle
        puzzle_dim: int = int(math.sqrt(int(puzzle_n_regex.group(1)) + 1))
        env = NPuzzle(puzzle_dim)
    elif 'lightsout' in env_name:
        from environments.lights_out import LightsOut
        m = re.search('lightsout([\d]+)', env_name)
        env = LightsOut(int(m.group(1)))
    elif env_name == 'sokoban':
        from environments.sokoban import Sokoban
        env = Sokoban(10, 4)
    elif env_name == 'cube2x2':
        from environments.cube2x2 import Cube2x2
        
        df_info_path = kwargs.get('df_info_path', '/mnt/hdd/santa/data/puzzle_info.csv')
        df_puzzles_path = kwargs.get('df_puzzles_path', '/mnt/hdd/santa/data/puzzles.csv')
        env = Cube2x2(df_info_path, df_puzzles_path)
    elif env_name == 'cube3x3':
        from environments.cube3x3 import Cube3x3
        
        df_info_path = kwargs.get('df_info_path', '/mnt/hdd/santa/data/puzzle_info.csv')
        df_puzzles_path = kwargs.get('df_puzzles_path', '/mnt/hdd/santa/data/puzzles.csv')
        env = Cube3x3(df_info_path, df_puzzles_path)
    elif env_name == 'cube4x4':
        from environments.cube4x4 import Cube4x4
        
        df_info_path = kwargs.get('df_info_path', '/mnt/hdd/santa/data/puzzle_info.csv')
        df_puzzles_path = kwargs.get('df_puzzles_path', '/mnt/hdd/santa/data/puzzles.csv')
        env = Cube4x4(df_info_path, df_puzzles_path)
    elif env_name == 'wreath21x21':
        from environments.wreath21x21 import Wreath21x21
        
        df_info_path = kwargs.get('df_info_path', '/mnt/hdd/santa/data/puzzle_info.csv')
        df_puzzles_path = kwargs.get('df_puzzles_path', '/mnt/hdd/santa/data/puzzles.csv')
        env = Wreath21x21(df_info_path, df_puzzles_path)
    elif env_name == 'wreath33x33':
        from environments.wreath33x33 import Wreath33x33
        
        df_info_path = kwargs.get('df_info_path', '/mnt/hdd/santa/data/puzzle_info.csv')
        df_puzzles_path = kwargs.get('df_puzzles_path', '/mnt/hdd/santa/data/puzzles.csv')
        env = Wreath33x33(df_info_path, df_puzzles_path)
    elif env_name == 'wreath100x100':
        from environments.wreath100x100 import Wreath100x100
        
        df_info_path = kwargs.get('df_info_path', '/mnt/hdd/santa/data/puzzle_info.csv')
        df_puzzles_path = kwargs.get('df_puzzles_path', '/mnt/hdd/santa/data/puzzles.csv')
        env = Wreath100x100(df_info_path, df_puzzles_path)
    else:
        raise ValueError('No known environment %s' % env_name)

    return env
