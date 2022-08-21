
from kaggle_environments.envs.kore_fleets.helpers import *
<<<<<<< Updated upstream
from extra_helpers import *
=======
from other_agents.extra_helpers import *
>>>>>>> Stashed changes

def should_mine(shipyard, best_fleet_size):
    if shipyard.ship_count >= best_fleet_size:
        return True
    return False

def check_path(board, start, dirs, dist_a, dist_b, collection_rate, L=False):
    kore = 0
    npv = .99
    current = start
    steps = 2 * (dist_a + dist_b + 2)
    for idx, d in enumerate(dirs):
        if L and idx==2:
            break
        for _ in range((dist_a if idx % 2 == 0 else dist_b) + 1):
            current = current.translate(d.to_point(), board.configuration.size)
            kore += int((board.cells.get(current).kore or 0) * collection_rate)
            final_kore = int((board.cells.get(current).kore or 0) * collection_rate)
    if L: kore = (kore) + (kore*(1-collection_rate)) - final_kore
    return math.pow(npv, steps) * kore / steps

def get_circular_flight_plan(gap1, gap2, start_dir):
    flight_plan = Direction.list_directions()[start_dir].to_char()
    if int(gap1):
        flight_plan += gap1
    next_dir = (start_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap2):
        flight_plan += gap2
    next_dir = (next_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap1):
        flight_plan += gap1
    next_dir = (next_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    return flight_plan

def get_L_flight_plan(gap1, gap2, start_dir):
    flight_plan = Direction.list_directions()[start_dir].to_char()
    if int(gap1):
        flight_plan += gap1
    next_dir = (start_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap2):
        flight_plan += gap2
    next_dir = (next_dir + 2) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap2):
        flight_plan += gap2
    next_dir = (next_dir - 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    return flight_plan

def get_rectangle_flight_plan(gap, start_dir):
    flight_plan = Direction.list_directions()[start_dir].to_char()
    next_dir = (start_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap):
        flight_plan += gap
    next_dir = (next_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    next_dir = (next_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    return flight_plan

def check_flight_paths(board, shipyard, search_radius):
    best_h = 0
    best_gap1 = 1
    best_gap2 = 1
    best_dir = board.step % 4
    for i in range(4):
        dirs = Direction.list_directions()[i:] + Direction.list_directions()[:i]
        for gap1 in range(0, search_radius):
            for gap2 in range(0, search_radius):
                fleet_size = min_ship_count_for_flight_plan_len(7)
                h = check_path(board, shipyard.position, dirs, gap1, gap2, collection_rate_for_ship_count(fleet_size), L=False)
                if h/fleet_size > best_h:
                    best_h = h/fleet_size
                    best_flight_plan = get_circular_flight_plan(str(gap1), str(gap2), i)
                    best_fleet_size = fleet_size
                h = check_path(board, shipyard.position, dirs, gap1, gap2, collection_rate_for_ship_count(collection_rate_for_ship_count(fleet_size)), L=True)
                if h/fleet_size > best_h:
                    best_h = h/fleet_size
                    best_flight_plan = get_L_flight_plan(str(gap1), str(gap2), i)
                    best_fleet_size = fleet_size
                if gap1!=0:
                    continue
                fleet_size = min_ship_count_for_flight_plan_len(5)
                h = check_path(board, shipyard.position, dirs, gap1, gap2, collection_rate_for_ship_count(fleet_size), L=False)
                if h/fleet_size > best_h:
                    best_h = h/fleet_size
                    best_flight_plan = get_rectangle_flight_plan(str(gap2), i)
                    best_fleet_size = fleet_size    
    return best_fleet_size, best_flight_plan                   
