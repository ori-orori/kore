
from kaggle_environments.envs.kore_fleets.helpers import *

def should_build(shipyard, remaining_kore):
    if remaining_kore > 500 and shipyard.max_spawn > 5:
        return True
    return False

def check_location(board, loc, me):
    if board.cells.get(loc).shipyard and board.cells.get(loc).shipyard.player.id == me.id:
        return 0
    kore = 0
    for i in range(-6, 7):
        for j in range(-6, 7):
            pos = loc.translate(Point(i, j), board.configuration.size)
            kore += board.cells.get(pos).kore or 0
    return kore

def build_new_shipyard(shipyard, board, me, convert_cost, search_radius=3):
    best_dir = 0
    best_kore = 0
    best_gap1 = 0
    best_gap2 = 0
    for i in range(4):
        next_dir = (i + 1) % 4
        for gap1 in range(0, search_radius, 1):
            for gap2 in range(0, search_radius, 1):
                enemy_shipyard_close = False
                diff1 = Direction.from_index(i).to_point() * gap1
                diff2 = Direction.from_index(next_dir).to_point() * gap2
                diff = diff1 + diff2
                pos = shipyard.position.translate(diff, board.configuration.size)
                for shipyard in board.shipyards.values():
                    if ((shipyard.player_id != me.id)
                        and (pos.distance_to(shipyard.position, board.configuration.size) < 4)):
                        enemy_shipyard_close = True
                if enemy_shipyard_close:
                    continue
                h = check_location(board, pos, me)
                if h > best_kore:
                    best_kore = h
                    best_gap1 = gap1
                    best_gap2 = gap2
                    best_dir = i
    gap1 = str(best_gap1)
    gap2 = str(best_gap2)
    next_dir = (best_dir + 1) % 4
    flight_plan = Direction.list_directions()[best_dir].to_char() + gap1
    flight_plan += Direction.list_directions()[next_dir].to_char() + gap2
    flight_plan += "C"
    return ShipyardAction.launch_fleet_with_flight_plan(max(convert_cost + 30, int(shipyard.ship_count/2)), flight_plan)
