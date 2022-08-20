from kaggle_environments import make

from kaggle_environments import make
env = make("kore_fleets", debug=True)
print(env.name, env.version)

def make_json(agent1, agent2, state):
    """
    
    """
    state_dict = {}
    agent1_action_dict = {}
    agent2_action_dict = {}

    if agent1 == agent and agent2 == agent:
        file_name = "baseline_vs_baseline.json"
        agent1_action_file_name = "baseline_action" + file_name
        agent2_action_file_name = "baseline_action" + file_name
    elif agent1 == agent:
        file_name = "baseline_vs_" + agent2[18:-3] + ".json"
        agent1_action_file_name = "baseline_action_" + file_name
        agent2_action_file_name = agent2[18:-3] + "_action_" + file_name
    elif agent2 == agent:
        file_name = agent1[18:-3] + "_vs_baseline" + ".json"
        agent1_action_file_name = agent1[18:-3] + "_action_" + file_name
        agent2_action_file_name = "baseline_action_" + file_name
    else:
        file_name = agent1[18:-3] + "_vs_" + agent2[18:-3] + ".json"
        agent1_action_file_name = agent1[18:-3] + "_action_" + file_name
        agent2_action_file_name = agent2[18:-3] + "_action_" + file_name
    

    for t in range(0,len(state)):
        state_dict[t+1] = state[t][0]['observation']
        agent1_action_dict[t+1] = state[t][0]['action']
        agent2_action_dict[t+1] = state[t][1]['action']

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent='\t')
    with open(agent1_action_file_name, 'w', encoding='utf-8') as f:
        json.dump(agent1_action_dict, f, indent='\t')
    with open(agent2_action_file_name, 'w', encoding='utf-8') as f:
        json.dump(agent2_action_dict, f, indent='\t')
    

from kaggle_environments.envs.kore_fleets.starter_bots.python.main import agent
import json
agents = ["kore/other_agents/beta_1st.py", "kore/other_agents/beta_6th.py", agent]



for idx1 in range(len(agents)):
    for idx2 in range(idx1, len(agents)):
        state = env.run([agents[idx1], agents[idx2]])
        
        make_json(agents[idx1], agents[idx2], state)
