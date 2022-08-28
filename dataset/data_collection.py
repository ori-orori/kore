import sys
import os
import json
import getopt
import yaml
import zipfile
import importlib.util
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.env_wrapper import KoreGymEnv


FILE_NAME = sys.argv[0]
MODE = "unzip"
CONFIG = os.path.abspath('..') + "/config/config.yaml"
PATH = os.path.abspath('..') + '/data/data.zip'

try:
    opts, etc_args = getopt.getopt(sys.argv[1:], "hm:c:p:", ["help", "mode=", "config=", "path="])

except getopt.GetoptError: # 옵션지정이 올바르지 않은 경우
    print(FILE_NAME, '-m <mode> -c <config file> -p <zipfile path>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print(FILE_NAME, '-m <mode> -c <config file> -p <zipfile path>')
        sys.exit()

    elif opt in ("-m", "--mode"):
        MODE = arg

    elif opt in ("-c", "--config"):
        CONFIG = arg

    elif opt in ("-p", "--path"):
        PATH = arg


with open(CONFIG) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if MODE == 'make':
    spec = importlib.util.spec_from_file_location(config['agents']['main_agent'],\
                                                  os.path.abspath('..') + '/other_agents/' + config['agents']['main_agent'] + '.py')
    my_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_agent)
    agent_idx = 0

    try:
        os.mkdir(os.path.abspath('..') + "/data/" + config['agents']['main_agent'])
    except:
        None
    file_path = os.path.abspath('..') + "/data/" + config['agents']['main_agent'] + '/'

    for opponent in config['agents']['opponent']:
        try:
            os.mkdir(file_path + opponent)
        except:
            None

    for opponent in config['agents']['opponent']:
        spec = importlib.util.spec_from_file_location(opponent, os.path.abspath('..') + '/other_agents/' + opponent + '.py')
        opponent_agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(opponent_agent)
        sample = config['agents']['samples_num'][agent_idx]

        for i in tqdm(range(sample)):
            kore_env = KoreGymEnv(config=dict(randomSeed=997269658))
            kore_env.run([my_agent.agent, opponent_agent.agent])
            env_json = kore_env.toJSON()

            file_name = file_path + opponent + '/' + str(i).zfill(5) + '.json'
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(env_json, f, indent='\t')

        agent_idx += 1


elif MODE == 'unzip':
    directory_to_extract_to = os.path.abspath('..') + '/data'

    with zipfile.ZipFile(PATH, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
