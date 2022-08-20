
from kaggle_environments.envs.kore_fleets.helpers import *

def should_defend(board, me, shipyard, radius=7):
    loc = shipyard.position
    for i in range(1-radius, radius):
        for j in range(1-radius, radius):
            pos = loc.translate(Point(i, j), board.configuration.size)
            if ((board.cells.get(pos).fleet is not None) 
                and (board.cells.get(pos).fleet.ship_count > 50)
                and (board.cells.get(pos).fleet.player_id!=me.id)
                and ((board.cells.get(pos).fleet.ship_count) > shipyard.ship_count)):
                return True               
    return False
