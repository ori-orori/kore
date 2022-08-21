import sys
import os
import json
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.starter_bots.python import main
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.env_wrapper import KoreGymEnv
from other_agents import beta_1st, beta_6th



agents = [beta_1st.agent, beta_6th.agent, main.agent]
agents_name = ["1st", "6th", "base"]
agent_idx = 0
sample_num = 10
file_path = os.path.abspath('..') + "/data/"

for agent in agents_name:
    try:
        os.mkdir(file_path + agent)
    except:
        None



for opponent in agents:
    opponent_name = agents_name[agent_idx]
    file_path = os.path.abspath('..') + '/data/' + opponent_name + '/'

    for i in range(sample_num):
        kore_env = KoreGymEnv(config=dict(randomSeed=997269658))
        kore_env.run([beta_1st.agent, opponent])
        env_json = kore_env.toJSON()

        file_name = file_path + str(i).zfill(2) + '.json'
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(env_json, f, indent='\t')

    agent_idx += 1

