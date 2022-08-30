import sys
import os
import json
from kaggle_environments import make
import torch
from torchvision import transforms, datasets
from env_wrapper import KoreGymEnv

file_path = os.path.abspath('..') + "/data/"
sample_num = 10
json_files = []

agents_name = ["1st", "6th", "base"]
for opponent in agents_name:
    for i in range(sample_num):
        json_path = file_path + opponent + '/' + str(i).zfill(5) + '.json'
        json_files.append(json_path)


class CustomDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.self_info = []
        self.map_info = []
        self.scalar_info = []
        self.actions = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):

        for json_file in self.files:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            for step in range(len(json_data['steps'])):
                observation = json_data['steps'][step][0]['observation']
                action = json_data['steps'][step][0]['action']
                states_per_step = KoreGymEnv.raw_obs_as_gym_state(observation)
                for single_state in states_per_step:
                    self.map_info.append(torch.tensor(single_state[0]))
                    self.scalar_info.append(torch.tensor(single_state[1]))
                    self.self_info.append(torch.tensor(single_state[2]))

                for shipyard in observation['players'][0][1]:
                    if shipyard in action:
                        self.actions.append(torch.tensor(KoreGymEnv.env_action_as_gym_action(action[shipyard])))
                    else:
                        self.actions.append(torch.tensor([0]))

        return self.map_info[i], self.scalar_info[i], self.self_info[i], self.actions[i]
