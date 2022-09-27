import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import yaml
import gym
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Board, Direction

env_config_path = './config/env_config.yaml'
with open(env_config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# setting constant
N_FEATURES = cfg['cell_features_dim']
MAX_EPISODE_STEPS = cfg['max_episode_steps']
MAX_OBSERVABLE_KORE = cfg['max_observable_kore']
MAX_FLEET_SHIPS = cfg['max_fleet_ships']
MAX_FLEET_KORE = cfg['max_fleet_kore']
MAX_OVERLAP_FLEETS = cfg['max_overlap_fleets']
MAX_SHIPYARD_SHIPS = cfg['max_shipyard_ships']
MAX_KORE_IN_RESERVE = cfg['max_kore_in_reserve']
FLIGHT_DISCOUNT_FACTOR = cfg['flight_discount_factor']
MAX_LAUNCH_SHIPS = cfg['max_launch_ships']
MAX_BUILD_SHIPS = cfg['max_build_ships']
MAX_FLIGHT_PLAN_INT = cfg['max_flight_plan_int']

class KoreGymEnv(gym.Env):
    CONFIG = make("kore_fleets").configuration
    def __init__(self, config=None, agents=None, env=None):
        super(KoreGymEnv, self).__init__()

        self.config = config
        self.agents = agents
        if env is None:
            self.env = make("kore_fleets", configuration=config)
        else:
            self.env = env
            
        self.config = self.env.configuration
        self.trainer = None
        self.raw_obs = None


    def step(self, action):
        """Execute action in the trainer and return the results.

        Args:
            action: The action in action space, i.e. the output of the our agent

        Returns:
            self.obs_as_gym_state: the current observation encoded as a state in state space
            reward: The agent's reward, 0 if not done else [+1 or -1]
            done: If True, the episode is over
            info: A dictionary with additional debugging information
        """
        kore_action = self.gym_to_kore_action(action)
        self.raw_obs, _, done, info = self.trainer.step(kore_action)  # Ignore trainer reward, which is just delta kore
        if done:
            player = int(self.raw_obs.player)
            opponent = 1 - player
            agent_reward = self.raw_obs["players"][player][0]
            opponent_reward = self.raw_obs["players"][opponent][0]
            self.reward = 1.0 if agent_reward > opponent_reward else -1.0
        else:
            self.reward = 0.0

        return self.obs_as_gym_state, self.reward, done, info

    def reset(self, agents=None):
        """Reset environment
        
        Returns:
            self.obs_as_gym_state: the first observation encoded as a state in state space
        """
        if agents is not None:
            self.agents = agents
        self.trainer = self.env.train(self.agents)
        self.raw_obs = self.trainer.reset()
        return self.obs_as_gym_state

    def render(self, **kwargs):
        self.env.render(**kwargs)

    @property
    def obs_as_gym_state(self) -> np.ndarray:
        """Return the current observation encoded as a state in state space.

        Define a 6x21x21+3 + 4 state (n_features x size x size , 3 extra features, 4 controlled shipyard features).
        # Feature 0: How much kore there is in a cell
        # Feature 1: How many ships there are in each fleet (>0: friendly, <0: enemy)
        # Feature 2: How much kore there is in each fleet
        # Feature 3: Where the friendly fleet will reach according to the flight plan
        # Feature 4: Where the enemy fleet will reach according to the flight plan
        # Feature 5: How many ships there are in each shipyard (>0: friendly, <0: enemy)
        # Feature 6: Progress - What turn is it?
        # Feature 7: How much kore do I have?
        # Feature 8: How much kore does the enemy have?
        # Feature 9: Controlled fleet x_position, y_position, ship count, id
        """
        # Init output state
        gym_state = np.zeros(shape=(N_FEATURES, self.config.size, self.config.size))

        # Get our player ID
        board = self.board
        our_id = board.current_player_id
        for point, cell in board.cells.items():
            # Feature 0: How much kore
            gym_state[0, point.y, point.x] = cell.kore

            # Feature 1: How many ships in each fleet (>0: friendly, <0: enemy)
            # Feature 2: How much kore in each fleet (>0: friendly, <0: enemy)
            fleet = cell.fleet
            if fleet:
                modifier = 1 if fleet.player_id == our_id else -1
                gym_state[1, point.y, point.x] = modifier * fleet.ship_count
                gym_state[2, point.y, point.x] = modifier * fleet.kore
            else:
                # The current cell has no fleet
                gym_state[1, point.y, point.x] = gym_state[2, point.y, point.x] = 0

            # Feature 3: Where the friendly fleet will reach
            # Feature 4: Where the enemy fleet will reach
            if fleet:
                modifier = 1 if fleet.player_id == our_id else -1
                current_position_y = point.y
                current_position_x = point.x
                current_direction = fleet.direction.to_char()
                step = 0
                if modifier == 1:
                    gym_state[3, current_position_y, current_position_x] \
                        += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                elif modifier == -1:
                    gym_state[4, current_position_y, current_position_x] \
                        += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                step += 1
                for char in fleet.flight_plan + ' ':
                    if char == 'C':
                        break
                    elif char in 'NEWS':
                        if char == 'N':
                            current_direction = 'N'
                            current_position_y = current_position_y - 1
                            current_position_x = current_position_x
                        elif char == 'E':
                            current_direction = 'E'
                            current_position_y = current_position_y
                            current_position_x = current_position_x + 1
                        elif char == 'W':
                            current_direction = 'W'
                            current_position_y = current_position_y
                            current_position_x = current_position_x - 1
                        elif char == 'S':
                            current_direction = 'S'
                            current_position_y = current_position_y + 1
                            current_position_x = current_position_x
                        current_position_y %= self.config.size
                        current_position_x %= self.config.size
                        if modifier == 1:
                            gym_state[3, current_position_y, current_position_x] \
                                += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                        elif modifier == -1:
                            gym_state[4, current_position_y, current_position_x] \
                                += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                        step += 1
                    elif char in '0123456789':
                        for _ in range(int(char)):
                            if step > 20:
                                break
                            elif current_direction == 'N':
                                current_position_y = current_position_y - 1
                                current_position_x = current_position_x
                            elif current_direction == 'E':
                                current_position_y = current_position_y
                                current_position_x = current_position_x + 1
                            elif current_direction == 'W':
                                current_position_y = current_position_y
                                current_position_x = current_position_x - 1
                            elif current_direction == 'S':
                                current_position_y = current_position_y + 1
                                current_position_x = current_position_x
                            current_position_y %= self.config.size
                            current_position_x %= self.config.size
                            if modifier == 1:
                                gym_state[3, current_position_y, current_position_x] \
                                    += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                            elif modifier == -1:
                                gym_state[4, current_position_y, current_position_x] \
                                    += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                            step += 1
                    elif char == ' ':
                        while step <= 20:
                            if current_direction == 'N':
                                current_position_y = current_position_y - 1
                                current_position_x = current_position_x
                            elif current_direction == 'E':
                                current_position_y = current_position_y
                                current_position_x = current_position_x + 1
                            elif current_direction == 'W':
                                current_position_y = current_position_y
                                current_position_x = current_position_x - 1
                            elif current_direction == 'S':
                                current_position_y = current_position_y + 1
                                current_position_x = current_position_x
                            current_position_y %= self.config.size
                            current_position_x %= self.config.size
                            if modifier == 1:
                                gym_state[3, current_position_y, current_position_x] \
                                    += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                            elif modifier == -1:
                                gym_state[4, current_position_y, current_position_x] \
                                    += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                            step += 1

                    if step > 20:
                        break
                    
            # Feature 5: How many ships in each shipyard (>0: friendly, <0: enemy)
            shipyard = cell.shipyard
            if shipyard:
                gym_state[5, point.y, point.x] =\
                    shipyard.ship_count if shipyard.player_id == our_id else -1 * shipyard.ship_count
            else:
                # The current cell has no shipyard
                gym_state[5, point.y, point.x] = 0

        me = board.current_player
        controlled_shipyard_info = None
        for shipyard in me.shipyards:
            shipyard_id = clip_normalize(int(shipyard.id.split('-')[0]), low_in=0, high_in=self.config.episodeSteps)
            shipyard_x_position = clip_normalize(shipyard.position[0], low_in=0, high_in=self.config.size)
            shipyard_y_position = clip_normalize(shipyard.position[1], low_in=0, high_in=self.config.size)
            shipyard_ship_count = clip_normalize(shipyard.ship_count, low_in=0, high_in=MAX_SHIPYARD_SHIPS)
            if controlled_shipyard_info is None:
                controlled_shipyard_info = np.array([[shipyard_id, shipyard_x_position, shipyard_y_position, shipyard_ship_count]])
            else:
                controlled_shipyard_info = np.concatenate((controlled_shipyard_info, 
                    np.array([[shipyard_id, shipyard_x_position, shipyard_y_position, shipyard_ship_count]])))
        # Normalize features
        # Feature 0: kore in range [0, MAX_OBSERVABLE_KORE]
        gym_state[0, :, :] = clip_normalize(
            x=gym_state[0, :, :], low_in=0, high_in=MAX_OBSERVABLE_KORE)
        # Feature 1: Ships on fleets in range [-MAX_FLEET_SHIPS, MAX_FLEET_SHIPS]
        gym_state[1, :, :] = clip_normalize(
            x=gym_state[1, :, :], low_in=-MAX_FLEET_SHIPS, high_in=MAX_FLEET_SHIPS, low_out=-1.0, high_out=1.0)

        # Feature 2:  Kore on fleets in range [-MAX_FLEET_KORE, MAX_FLEET_KORE]
        gym_state[2, :, :] = clip_normalize(
            x=gym_state[2, :, :], low_in=-MAX_FLEET_KORE, high_in=MAX_FLEET_KORE, low_out=-1.0, high_out=1.0)

        # Feature 3: Point of friendly fleet will reach in range [0, MAX_OVERLAP_FLEETS]
        gym_state[3, :, :] = clip_normalize(
            x=gym_state[3, :, :], low_in=0, high_in=MAX_OVERLAP_FLEETS)
        
        # Feature 4: Point of enemy fleet will reach in range [-MAX_OVERLAP_FLEETS, 0]
        gym_state[4, :, :] = clip_normalize(
            x=gym_state[4, :, :], low_in=-MAX_OVERLAP_FLEETS, high_in=0, low_out=-1.0, high_out=0.0)
        
        # Feature 5: ships on shipyard in range [-MAX_SHIPYARD_SHIPS, MAX_SHIPYARD_SHIPS]
        gym_state[5, :, :] = clip_normalize(
            x=gym_state[5, :, :], low_in=-MAX_SHIPYARD_SHIPS, high_in=MAX_SHIPYARD_SHIPS, low_out=-1.0, high_out=1.0)
        
        # Extra Features: Progress, how much kore do I have, how much kore does opponent have
        player = board.current_player
        opponent = board.opponents[0]
        progress = clip_normalize(
            board.step, low_in=0, high_in=MAX_EPISODE_STEPS, low_out=0, high_out=1)
        my_kore = clip_normalize(player.kore, low_in=0, high_in=MAX_KORE_IN_RESERVE)
        opponent_kore = clip_normalize(opponent.kore, low_in=0, high_in=MAX_KORE_IN_RESERVE)
        states = []
        for shipyard_info in controlled_shipyard_info:
            states.append([gym_state, np.array([progress, my_kore, opponent_kore]), shipyard_info]) 
        return states

    @property
    def board(self):
        return Board(self.raw_obs, self.config)

    def gym_to_kore_action(self, gym_action):
        """Decode an action in action space as a kore action.

        We will interpret the values as follows:
        if gym_action[0] > 0 launch a fleet, elif < 0 build ships, else wait.
        abs(gym_action[0]) encodes the number of ships to build/launch.
        gym_action[2k+1] represents the direction in which to launch the fleet.
        gym_action[2k+2] represents the step in which to launch the fleet.
        k = 0, ..., n-1. n pairs of direction and step

        Args:
            gym_action: The action produces by our agent.

        Returns:
            The corresponding kore environment actions or None if the agent wants to wait.

        """
        # Broadcast the same action to all shipyards
        board = self.board
        me = board.current_player
        for shipyard in me.shipyards:
            shipyard_id = shipyard.id.split('-')[0]
            action_launch = gym_action[shipyard_id][0] > 0
            action_build = gym_action[shipyard_id][0] < 0

            if action_launch:
                number_of_ships = int(
                    clip_normalize(
                        x=abs(gym_action[shipyard_id][0]),
                        low_in=0,
                        high_in=1,
                        low_out=1,
                        high_out=MAX_LAUNCH_SHIPS,
                    )
                )
            elif action_build:
                number_of_ships = int(
                    clip_normalize(
                        x=abs(gym_action[shipyard_id][0]),
                        low_in=0,
                        high_in=1,
                        low_out=1,
                        high_out=MAX_BUILD_SHIPS,
                    )
                )

            action = None
            if action_build:
                max_spawn = shipyard.max_spawn
                number_of_ships = min(number_of_ships, max_spawn)
                if number_of_ships:
                    action = ShipyardAction.spawn_ships(number_ships=number_of_ships)

            elif action_launch:
                shipyard_count = shipyard.ship_count
                number_of_ships = min(number_of_ships, shipyard_count)
                if number_of_ships:
                    flight_plan = ""
                    for k in range(len(gym_action[shipyard_id]) // 2):
                        direction = int(gym_action[shipyard_id][2*k+1])  # int between 0 (North) and 3 (West), 4(Build shipyard)
                        if direction == 0:
                            direction = 'N'
                        elif direction == 1:
                            direction = 'E'
                        elif direction == 2:
                            direction = 'S'
                        elif direction == 3:
                            direction = 'W'
                        elif direction == 4:
                            direction = 'C'
                        step = int(
                            clip_normalize(
                                x=abs(gym_action[shipyard_id][2*k+2]),
                                low_in=0,
                                high_in=1,
                                low_out=0,
                                high_out=MAX_FLIGHT_PLAN_INT,
                            )
                        )

                        flight_plan = flight_plan + direction + str(step)
                    action = ShipyardAction.launch_fleet_with_flight_plan(number_ships=number_of_ships, 
                                    flight_plan=flight_plan)
            shipyard.next_action = action
            
        return me.next_actions

    def toJSON(self):
        return self.env.toJSON()

    def run(self, agents=None):
        """
        
        Args:
            agents : ['./other_agents/beta_1st.py', './other_agents/beta_6th.py']
        """
        if agents is not None:
            self.agents = agents
        return self.env.run(self.agents)

    @staticmethod
    def raw_obs_as_gym_state(raw_obs, config) -> np.ndarray:
        """Return the current observation encoded as a state in state space.

        Define a 6x21x21+3+4 state (n_features x size x size x, 3 extra features and 4 controlled shipyard features).
        # Feature 0: How much kore there is in a cell
        # Feature 1: How many ships there are in each fleet (>0: friendly, <0: enemy)
        # Feature 2: How much kore there is in each fleet
        # Feature 3: Where the friendly fleet will reach according to the flight plan
        # Feature 4: Where the enemy fleet will reach according to the flight plan
        # Feature 5: How many ships there are in each shipyard (>0: friendly, <0: enemy)
        # Feature 6: Progress - What turn is it?
        # Feature 7: How much kore do I have?
        # Feature 8: How much kore does the enemy have?
        # Feature 9: Controlled fleet x_position, y_position, ship count, id

        Args:
            raw_obs: raw observation of kore environment 

        """
        # Init output state
        gym_state = np.zeros(shape=(N_FEATURES, config.size, config.size))

        # Get our player ID
        board = Board(raw_obs, config)
        our_id = board.current_player_id

        for point, cell in board.cells.items():
            # Feature 0: How much kore
            gym_state[0, point.y, point.x] = cell.kore

            # Feature 1: How many ships in each fleet (>0: friendly, <0: enemy)
            # Feature 2: How much kore in each fleet (>0: friendly, <0: enemy)
            fleet = cell.fleet
            if fleet:
                modifier = 1 if fleet.player_id == our_id else -1
                gym_state[1, point.y, point.x] = modifier * fleet.ship_count
                gym_state[2, point.y, point.x] = modifier * fleet.kore
            else:
                # The current cell has no fleet
                gym_state[1, point.y, point.x] = gym_state[2, point.y, point.x] = 0

            # Feature 3: Where the friendly fleet will reach
            # Feature 4: Where the enemy fleet will reach
            if fleet:
                modifier = 1 if fleet.player_id == our_id else -1
                current_position_y = point.y
                current_position_x = point.x
                current_direction = fleet.direction.to_char()
                step = 0
                if modifier == 1:
                    gym_state[3, current_position_y, current_position_x] \
                        += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                elif modifier == -1:
                    gym_state[4, current_position_y, current_position_x] \
                        += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                step += 1
                for char in fleet.flight_plan + ' ':
                    if char == 'C':
                        break
                    elif char in 'NEWS':
                        if char == 'N':
                            current_direction = 'N'
                            current_position_y = current_position_y - 1
                            current_position_x = current_position_x
                        elif char == 'E':
                            current_direction = 'E'
                            current_position_y = current_position_y
                            current_position_x = current_position_x + 1
                        elif char == 'W':
                            current_direction = 'W'
                            current_position_y = current_position_y
                            current_position_x = current_position_x - 1
                        elif char == 'S':
                            current_direction = 'S'
                            current_position_y = current_position_y + 1
                            current_position_x = current_position_x
                        current_position_y %= config.size
                        current_position_x %= config.size
                        if modifier == 1:
                            gym_state[3, current_position_y, current_position_x] \
                                += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                        elif modifier == -1:
                            gym_state[4, current_position_y, current_position_x] \
                                += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                        step += 1
                    elif char in '0123456789':
                        for _ in range(int(char)):
                            if step > 20:
                                break
                            elif current_direction == 'N':
                                current_position_y = current_position_y - 1
                                current_position_x = current_position_x
                            elif current_direction == 'E':
                                current_position_y = current_position_y
                                current_position_x = current_position_x + 1
                            elif current_direction == 'W':
                                current_position_y = current_position_y
                                current_position_x = current_position_x - 1
                            elif current_direction == 'S':
                                current_position_y = current_position_y + 1
                                current_position_x = current_position_x
                            current_position_y %= config.size
                            current_position_x %= config.size
                            if modifier == 1:
                                gym_state[3, current_position_y, current_position_x] \
                                    += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                            elif modifier == -1:
                                gym_state[4, current_position_y, current_position_x] \
                                    += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                            step += 1
                    elif char == ' ':
                        while step <= 20:
                            if current_direction == 'N':
                                current_position_y = current_position_y - 1
                                current_position_x = current_position_x
                            elif current_direction == 'E':
                                current_position_y = current_position_y
                                current_position_x = current_position_x + 1
                            elif current_direction == 'W':
                                current_position_y = current_position_y
                                current_position_x = current_position_x - 1
                            elif current_direction == 'S':
                                current_position_y = current_position_y + 1
                                current_position_x = current_position_x
                            current_position_y %= config.size
                            current_position_x %= config.size
                            if modifier == 1:
                                gym_state[3, current_position_y, current_position_x] \
                                    += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                            elif modifier == -1:
                                gym_state[4, current_position_y, current_position_x] \
                                    += modifier * 1.0 * (FLIGHT_DISCOUNT_FACTOR ** step)
                            step += 1

                    if step > 20:
                        break
                    
            # Feature 5: How many ships in each shipyard (>0: friendly, <0: enemy)
            shipyard = cell.shipyard
            if shipyard:
                gym_state[5, point.y, point.x] =\
                    shipyard.ship_count if shipyard.player_id == our_id else -1 * shipyard.ship_count
            else:
                # The current cell has no shipyard
                gym_state[5, point.y, point.x] = 0

        me = board.current_player
        controlled_shipyard_info = None
        for shipyard in me.shipyards:
            shipyard_id = clip_normalize(int(shipyard.id.split('-')[0]), low_in=0, high_in=config.episodeSteps)
            shipyard_x_position = clip_normalize(shipyard.position[0], low_in=0, high_in=config.size-1)
            shipyard_y_position = clip_normalize(shipyard.position[1], low_in=0, high_in=config.size-1)
            shipyard_ship_count = clip_normalize(shipyard.ship_count, low_in=0, high_in=MAX_SHIPYARD_SHIPS)
            if controlled_shipyard_info is None:
                controlled_shipyard_info = np.array([[shipyard_id, shipyard_x_position, shipyard_y_position, shipyard_ship_count]])
            else:
                controlled_shipyard_info = np.concatenate((controlled_shipyard_info, 
                    np.array([[shipyard_id, shipyard_x_position, shipyard_y_position, shipyard_ship_count]])))
        # Normalize features
        # Feature 0: kore in range [0, MAX_OBSERVABLE_KORE]
        gym_state[0, :, :] = clip_normalize(
            x=gym_state[0, :, :], low_in=0, high_in=MAX_OBSERVABLE_KORE)
        # Feature 1: Ships on fleets in range [-MAX_FLEET_SHIPS, MAX_FLEET_SHIPS]
        gym_state[1, :, :] = clip_normalize(
            x=gym_state[1, :, :], low_in=-MAX_FLEET_SHIPS, high_in=MAX_FLEET_SHIPS, low_out=-1.0, high_out=1.0)

        # Feature 2:  Kore on fleets in range [-MAX_FLEET_KORE, MAX_FLEET_KORE]
        gym_state[2, :, :] = clip_normalize(
            x=gym_state[2, :, :], low_in=-MAX_FLEET_KORE, high_in=MAX_FLEET_KORE, low_out=-1.0, high_out=1.0)

        # Feature 3: Point of friendly fleet will reach in range [0, MAX_OVERLAP_FLEETS]
        gym_state[3, :, :] = clip_normalize(
            x=gym_state[3, :, :], low_in=0, high_in=MAX_OVERLAP_FLEETS)
        
        # Feature 4: Point of enemy fleet will reach in range [-MAX_OVERLAP_FLEETS, 0]
        gym_state[4, :, :] = clip_normalize(
            x=gym_state[4, :, :], low_in=-MAX_OVERLAP_FLEETS, high_in=0, low_out=-1.0, high_out=0.0)
        
        # Feature 5: ships on shipyard in range [-MAX_SHIPYARD_SHIPS, MAX_SHIPYARD_SHIPS]
        gym_state[5, :, :] = clip_normalize(
            x=gym_state[5, :, :], low_in=-MAX_SHIPYARD_SHIPS, high_in=MAX_SHIPYARD_SHIPS, low_out=-1.0, high_out=1.0)
        
        # Extra Features: Progress, how much kore do I have, how much kore does opponent have
        player = board.current_player
        opponent = board.opponents[0]
        progress = clip_normalize(
            board.step, low_in=0, high_in=MAX_EPISODE_STEPS, low_out=0, high_out=1)
        my_kore = clip_normalize(player.kore, low_in=0, high_in=MAX_KORE_IN_RESERVE)
        opponent_kore = clip_normalize(opponent.kore, low_in=0, high_in=MAX_KORE_IN_RESERVE)
        states = []
        if controlled_shipyard_info is not None: 
            for shipyard_info in controlled_shipyard_info:
                states.append([gym_state, np.array([progress, my_kore, opponent_kore]), shipyard_info]) 
            
        return states

    @staticmethod
    def env_action_as_gym_action(action):
        direction_list = {'N' : 0, 'E' : 1, 'S' : 2, 'W' : 3, 'C' : 4}
        ppo_action = []
        act = action.split('_')

        if act[0]=='SPAWN':
            ppo_action = [1, int(act[1])] + [0]*10

        elif act[0] == 'LAUNCH':
            dir_step = []
            plan = act[2]
            for i in range(len(plan)):
                if plan[i] in direction_list:
                    if i==len(plan)-1 or plan[i+1] in direction_list:
                        dir_step.extend([direction_list[plan[i]], 0])
                    else:
                        dir_step.extend([direction_list[plan[i]], int(plan[i+1])/9])

            ppo_action = [1, int(act[1])]
            ppo_action.extend(dir_step)
            ppo_action += [0] * (12 - len(ppo_action))
        else:
            ppo_action = [0] * 12
        
        return np.array(ppo_action)

def clip_normalize(x, low_in, high_in, low_out=0.0, high_out=1.0):
    """Clip values in x to the interval [low_in, high_in] and then MinMax-normalize to [low_out, high_out].

    Args:
        x: The array of float to clip and normalize
        low_in: The lowest possible value in x
        high_in: The highest possible value in x
        low_out: The lowest possible value in the output
        high_out: The highest possible value in the output

    Returns:
        The clipped and normalized version of x

    Raises:
        AssertionError if the limits are not consistent

    Examples:
        >>> clip_normalize(50, low_in=0, high_in=100)
        0.0

        >>> clip_normalize(np.array([-1, .5, 99]), low_in=-1, high_in=1, low_out=0, high_out=2)
        array([0., 1.5, 2.])
    """
    assert high_in > low_in and high_out > low_out, "Wrong limits"

    # Clip outliers
    try:
        x[x > high_in] = high_in
        x[x < low_in] = low_in
    except TypeError:
        x = high_in if x > high_in else x
        x = low_in if x < low_in else x
    # y = ax + b
    a = (high_out - low_out) / (high_in - low_in)
    b = high_out - high_in * a

    return a * x + b

# test
# env = KoreGymEnv()
# obs = env.reset([None, '../other_agents/beta_1st.py'])
# obs = env.reset(['../other_agents/beta_1st.py', None])
# env.step({'0':[0]})
# env.step({'0':[0]})
# env.step({'0':[-0.3]})
# print(env.raw_obs)
# print(env.raw_obs["players"])
# print(dir(env.raw_obs))
# print(env.board)
# print(env.raw_obs.player)