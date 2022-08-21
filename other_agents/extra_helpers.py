
from kaggle_environments.envs.kore_fleets.helpers import *
from random import choice, randint, randrange, sample, seed, random

def get_col_row(size, pos):
    return pos % size, pos // size

def get_to_pos(size, pos, direction):
    col, row = get_col_row(size, pos)
    if direction == "NORTH":
        return pos - size if pos >= size else size ** 2 - size + col
    elif direction == "SOUTH":
        return col if pos + size >= size ** 2 else pos + size
    elif direction == "EAST":
        return pos + 1 if col < size - 1 else row * size
    elif direction == "WEST":
        return pos - 1 if col > 0 else (row + 1) * size - 1
    
def get_shortest_flight_path_between(position_a, position_b, size, trailing_digits=False):
    mag_x = 1 if position_b.x > position_a.x else -1
    abs_x = abs(position_b.x - position_a.x)
    dir_x = mag_x if abs_x < size/2 else -mag_x
    mag_y = 1 if position_b.y > position_a.y else -1
    abs_y = abs(position_b.y - position_a.y)
    dir_y = mag_y if abs_y < size/2 else -mag_y
    flight_path_x = ""
    if abs_x > 0:
        flight_path_x += "E" if dir_x == 1 else "W"
        flight_path_x += str(abs_x - 1) if (abs_x - 1) > 0 else ""
    flight_path_y = ""
    if abs_y > 0:
        flight_path_y += "N" if dir_y == 1 else "S"
        flight_path_y += str(abs_y - 1) if (abs_y - 1) > 0 else ""
    if not len(flight_path_x) == len(flight_path_y):
        if len(flight_path_x) < len(flight_path_y):
            return flight_path_x + (flight_path_y if trailing_digits else flight_path_y[0])
        else:
            return flight_path_y + (flight_path_x if trailing_digits else flight_path_x[0])
    return flight_path_y + (flight_path_x if trailing_digits or not flight_path_x else flight_path_x[0]) if random() < .5 else flight_path_x + (flight_path_y if trailing_digits or not flight_path_y else flight_path_y[0])

def get_total_ships(board, player):
    ships = 0
    for fleet in board.fleets.values():
        if fleet.player_id == player:
            ships += fleet.ship_count
    for shipyard in board.shipyards.values():
        if shipyard.player_id == player:
            ships += shipyard.ship_count
    return ships    

# ref @egrehbbt 
def max_flight_plan_len_for_ship_count(ship_count):
    return math.floor(2 * math.log(ship_count)) + 1

# ref @egrehbbt 
def min_ship_count_for_flight_plan_len(flight_plan_len):
    return math.ceil(math.exp((flight_plan_len - 1) / 2))

# ref @egrehbbt 
def collection_rate_for_ship_count(ship_count):
    return min(math.log(ship_count) / 20, 0.99)

def spawn_ships(shipyard, remaining_kore, spawn_cost):
    return ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))
