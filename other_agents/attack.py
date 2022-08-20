
from kaggle_environments.envs.kore_fleets.helpers import *

def should_attack(board, shipyard, remaining_kore, spawn_cost, invading_fleet_size):
            closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, board.current_player)
            dist_to_closest_enemy_shipyard = 100 if not closest_enemy_shipyard else shipyard.position.distance_to(closest_enemy_shipyard.position, board.configuration.size)
            if (closest_enemy_shipyard 
                and (closest_enemy_shipyard.ship_count < 20 or dist_to_closest_enemy_shipyard < 15) 
                and (remaining_kore >= spawn_cost or shipyard.ship_count >= invading_fleet_size) 
                and (board.step > 300 or dist_to_closest_enemy_shipyard < 12)):
                return True
            return False

def get_closest_enemy_shipyard(board, position, me):
    min_dist = 1000000
    enemy_shipyard = None
    for shipyard in board.shipyards.values():
        if shipyard.player_id == me.id:
            continue
        dist = position.distance_to(shipyard.position, board.configuration.size)
        if dist < min_dist:
            min_dist = dist
            enemy_shipyard = shipyard
    return enemy_shipyard

