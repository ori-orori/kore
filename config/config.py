import numpy as np
from kaggle_environments import make

# Read env specification
ENV_SPECIFICATION = make('kore_fleets').specification
SHIP_COST = ENV_SPECIFICATION.configuration.spawnCost.default
SHIPYARD_COST = ENV_SPECIFICATION.configuration.convertCost.default
GAME_CONFIG = {
    'episodeSteps':  ENV_SPECIFICATION.configuration.episodeSteps.default,  # You might want to start with smaller values
    'size': ENV_SPECIFICATION.configuration.size.default
}

# Define your opponent. We'll use the starter bot in the notebook environment for this baseline.
MAIN_AGENT = 'beta_1st'
OPPONENT = ['beta_1st', 'beta_6th', 'opponent']
SAMPLES_NUM = [1000, 500, 100]

# Define our parameters
N_FEATURES = 6
ACTION_SIZE = (2,)
DTYPE = np.float64
MAX_OBSERVABLE_KORE = 500
MAX_FLEET_SHIPS = 200
MAX_FLEET_KORE = 1000
MAX_OVERLAP_FLEETS = 10
MAX_SHIPYARD_SHIPS = 300
MAX_KORE_IN_RESERVE = 40000
FLIGHT_DISCOUNT_FACTOR = 0.99
MAX_LAUNCH_SHIPS = 100
MAX_BUILD_SHIPS = 10
MAX_FLIGHT_PLAN_INT = 9