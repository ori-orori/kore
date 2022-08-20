
from kaggle_environments.envs.kore_fleets.helpers import *
from extra_helpers import *
from defend import *
from attack import *
from build import *
from mine import *
from random import randint
import itertools
import numpy as np
from random import choice, randint, randrange, sample, seed, random
import math

def agent(obs, config):
    
    board = Board(obs, config)
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    turn = board.step
    
    invading_fleet_size = 75
    convert_cost_buffer = 80
    mining_search_radius = 10
    defence_radius = 7
    
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        
        best_fleet_size, best_flight_plan = check_flight_paths(board, shipyard, mining_search_radius) 
        
        if should_defend(board, me, shipyard, defence_radius):
            if remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                
        elif should_attack(board, shipyard, remaining_kore, spawn_cost, invading_fleet_size):
            if shipyard.ship_count >= invading_fleet_size:
                    closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, board.current_player)
                    flight_plan = get_shortest_flight_path_between(shipyard.position, closest_enemy_shipyard.position, size)
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(invading_fleet_size, flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)

        elif should_build(shipyard, remaining_kore):
            if shipyard.ship_count >= convert_cost + convert_cost_buffer:
                shipyard.next_action = build_new_shipyard(shipyard, board, me, convert_cost)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                
        elif should_mine(shipyard, best_fleet_size):
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(best_fleet_size, best_flight_plan)
        
        elif (remaining_kore > spawn_cost):
            shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
        
        elif (len(me.fleet_ids) == 0 and shipyard.ship_count <= 22) and len(shipyards)==1:
            if remaining_kore > 11:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
            else:
                direction = Direction.NORTH
                if shipyard.ship_count > 0:
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, direction.to_char())
                
    return me.next_actions
