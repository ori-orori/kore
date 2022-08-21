import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import json
import warnings
from kaggle_environments import make, Environment
from env_wrapper import KoreGymEnv


def render_replay_game(file_path, mode='ipython', width=750, height=750):
    """ Render game of saved json file
    
    Args:
        file_path : path of saved json file
    
    Returns:
        Ipython object of playing video framework
    """
    env = replay_game(file_path)
    warnings.simplefilter('ignore') # ignore warning messages
    return env.render(mode=mode, width=width, height=height)

def replay_game(file_path):
    """ Render game of saved json file
    
    Args:
        file_path : path of saved json file
    
    Returns:
        kore gym environment
    """
    kore_env = make("kore_fleets")

    with open(file_path, 'r') as file:
        data = json.load(file)

    # make new environment
    env = Environment(
            specification=data["specification"],
            configuration=data["configuration"],
            steps=data["steps"],
            agents=kore_env.agents,
            interpreter=kore_env.interpreter,
            renderer=kore_env.renderer,
            html_renderer=kore_env.html_renderer,
            debug=kore_env.debug,
        )
    return KoreGymEnv(env=env)