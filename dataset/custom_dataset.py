import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time
import json
from kaggle_environments import make
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from dataset.env_wrapper import KoreGymEnv

class KoreDataset(Dataset):
    def __init__(self, cfg):
        file_path = cfg['train']['sl']['dataset']['data_path']
        samples_num_list = cfg['agents']['samples_num']
        opponents = cfg['agents']['opponent']
        main = cfg['agents']['main_agent']
        self.files = []
        for opponent, samples_num in zip(opponents, samples_num_list):
            for i in range(samples_num):
                json_path = file_path + main + '/' + opponent + '/' + str(i).zfill(2) + '.json'
                self.files.append(json_path)

        self.self_info = []
        self.map_info = []
        self.scalar_info = []
        self.actions = []
        self.config = make("kore_fleets").configuration

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        for json_file in self.files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            for step in range(5): # len(json_data['steps'])
                observation = json_data['steps'][step][0]['observation']
                action = json_data['steps'][step][0]['action']             
                states_per_step = KoreGymEnv.raw_obs_as_gym_state(observation, self.config)
                if len(states_per_step) != 0:
                    for single_state in states_per_step:
                        self.map_info.append(torch.tensor(single_state[0]).to(torch.float32))
                        self.scalar_info.append(torch.tensor(single_state[1]).to(torch.float32))
                        self.self_info.append(torch.tensor(single_state[2]).to(torch.float32))

                    for shipyard in observation['players'][0][1]:
                        if shipyard in action:
                            self.actions.append(torch.tensor(KoreGymEnv.env_action_as_gym_action(action[shipyard])).to(torch.float32))
                        else:
                            self.actions.append(torch.tensor([0]*12, dtype=torch.float32))
        return (self.map_info[i], self.scalar_info[i], self.self_info[i]), self.actions[i]
