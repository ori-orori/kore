import math
import random
from typing import Dict, List, Union, Optional, Generator, Tuple
from collections import defaultdict
from kaggle_environments.envs.kore_fleets.helpers import Configuration
from kaggle_environments.envs.kore_fleets.helpers import SPAWN_VALUES
import numpy as np
#%%writefile expantion.py
from collections import defaultdict
import itertools


def max_ships_to_spawn(turns_controlled: int) -> int:
    for idx, target in enumerate(SPAWN_VALUES):
        if turns_controlled < target:
            return idx + 1
    return len(SPAWN_VALUES) + 1


def max_flight_plan_len_for_ship_count(ship_count: int) -> int:
    return math.floor(2 * math.log(ship_count)) + 1


def min_ship_count_for_flight_plan_len(flight_plan_len: int) -> int:
    return math.ceil(math.exp((flight_plan_len - 1) / 2))


def collection_rate_for_ship_count(ship_count: int) -> float:
    return min(math.log(ship_count) / 20, 0.99)


def create_spawn_ships_command(num_ships: int) -> str:
    return f"SPAWN_{num_ships}"


def create_launch_fleet_command(num_ships: int, plan: str) -> str:
    return f"LAUNCH_{num_ships}_{plan}"


class cached_property:
    """
    python 3.9:
    >>> from functools import cached_property
    """

    def __init__(self, func):
        self.func = func
        self.key = "__" + func.__name__

    def __get__(self, instance, owner):
        try:
            return instance.__getattribute__(self.key)
        except AttributeError:
            value = self.func(instance)
            instance.__setattr__(self.key, value)
            return value


class cached_call:
    """
    may cause a memory leak, be careful
    """

    def __init__(self, func):
        self.func = func
        self.key = "__" + func.__name__

    def __get__(self, instance, owner):
        try:
            d = instance.__getattribute__(self.key)
        except AttributeError:
            d = {}
            instance.__setattr__(self.key, d)

        def func(x):
            try:
                return d[x]
            except KeyError:
                value = self.func(instance, x)
                d[x] = value
                return value

        return func


class Obj:
    def __init__(self, game_id: Union[str, int]):
        self._game_id = game_id

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._game_id})"

    @property
    def game_id(self):
        return self._game_id

class Action(Obj):
    def __init__(self, dx, dy, game_id, command):
        super().__init__(game_id)
        self._dx = dx
        self._dy = dy
        self._command = command

    def __repr__(self):
        return self._command

    @property
    def dx(self) -> int:
        return self._dx

    @property
    def dy(self) -> int:
        return self._dy

    @property
    def command(self) -> str:
        return self._command


North = Action(
    dx=0,
    dy=1,
    game_id=0,
    command="N",
)
East = Action(
    dx=1,
    dy=0,
    game_id=1,
    command="E",
)
South = Action(
    dx=0,
    dy=-1,
    command="S",
    game_id=2,
)
West = Action(
    dx=-1,
    dy=0,
    command="W",
    game_id=3,
)
Convert = Action(
    dx=0,
    dy=0,
    command="C",
    game_id=-1,
)


ALL_DIRECTIONS = {North, East, South, West}
ALL_ACTIONS = {North, East, South, West, Convert}
GAME_ID_TO_ACTION = {x.game_id: x for x in ALL_ACTIONS}
COMMAND_TO_ACTION = {x.command: x for x in ALL_ACTIONS}
ACTION_TO_OPPOSITE_ACTION = {
    North: South,
    East: West,
    South: North,
    West: East,
}


def get_opposite_action(action):
    return ACTION_TO_OPPOSITE_ACTION.get(action, action)


class Point(Obj):
    def __init__(self, x: int, y: int, kore: float, field: "Field"):
        super().__init__(game_id=(field.size - y - 1) * field.size + x)
        self._x = x
        self._y = y
        self._kore = kore
        self._field = field

    def __repr__(self):
        return f"Point({self._x}, {self._y})"

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    def to_tuple(self) -> Tuple[int, int]:
        return self._x, self._y

    @property
    def kore(self) -> float:
        return self._kore

    def set_kore(self, kore: float):
        self._kore = kore

    @property
    def field(self) -> "Field":
        return self._field

    def apply(self, action: Action) -> "Point":
        return self._field[(self.x + action.dx, self.y + action.dy)]

    @cached_call
    def distance_from(self, point: "Point") -> int:
        return sum(p.num_steps for p in self.dirs_to(point))

    @cached_property
    def adjacent_points(self) -> List["Point"]:
        return [self.apply(a) for a in ALL_DIRECTIONS]

    @cached_property
    def row(self) -> List["Point"]:
        return list(self._field.points[:, self.y])

    @cached_property
    def column(self) -> List["Point"]:
        return list(self._field.points[self.x, :])

    @cached_call
    def nearby_points(self, r: int) -> List["Point"]:
        if r > 1:
            points = []
            for p in self._field:
                distance = self.distance_from(p)
                if 0 < distance <= r:
                    points.append(p)
            return points
        elif r == 1:
            return self.adjacent_points

        raise ValueError("Radius must be more or equal then 1")

    @cached_call
    def dirs_to(self, point: "Point") -> List["PlanPath"]:
        dx, dy = self._field.swap(self._x - point.x, self._y - point.y)
        ret = []
        if dx:
            ret.append(PlanPath(West, dx))
        if dy:
            ret.append(PlanPath(South, dy))
        return ret


class Field:
    def __init__(self, size: int):
        self._size = size
        self._points = self.create_array(size)

    def __iter__(self) -> Generator[Point, None, None]:
        for row in self._points:
            yield from row

    def create_array(self, size: int) -> np.ndarray:
        ar = np.zeros((size, size), dtype=Point)
        for x in range(size):
            for y in range(size):
                point = Point(x, y, kore=0, field=self)
                ar[x, y] = point
        return ar

    @property
    def points(self) -> np.ndarray:
        return self._points

    def get_row(self, y: int, start: int, size: int) -> List[Point]:
        if size < 0:
            return self.get_row(y, start=start + size + 1, size=-size)[::-1]

        ps = self._points
        start %= self._size
        out = []
        while size > 0:
            d = list(ps[slice(start, start + size), y])
            size -= len(d)
            start = 0
            out += d
        return out

    def get_column(self, x: int, start: int, size: int) -> List[Point]:
        if size < 0:
            return self.get_column(x, start=start + size + 1, size=-size)[::-1]

        ps = self._points
        start %= self._size
        out = []
        while size > 0:
            d = list(ps[x, slice(start, start + size)])
            size -= len(d)
            start = 0
            out += d
        return out

    @property
    def size(self) -> int:
        return self._size

    def swap(self, dx, dy):
        size = self._size
        if abs(dx) > size / 2:
            dx -= np.sign(dx) * size
        if abs(dy) > size / 2:
            dy -= np.sign(dy) * size
        return dx, dy

    def __getitem__(self, item) -> Point:
        x, y = item
        return self._points[x % self._size, y % self._size]


class PlanPath:
    def __init__(self, direction: Action, num_steps: int = 0):
        if direction == Convert:
            self._direction = direction
            self._num_steps = 0
        elif num_steps > 0:
            self._direction = direction
            self._num_steps = num_steps
        else:
            self._direction = get_opposite_action(direction)
            self._num_steps = -num_steps

    def __repr__(self):
        return self.to_str()

    @property
    def direction(self):
        return self._direction

    @property
    def num_steps(self):
        return self._num_steps

    def to_str(self):
        if self.direction == Convert:
            return Convert.command
        elif self.num_steps == 0:
            return ""
        elif self.num_steps == 1:
            return self.direction.command
        else:
            return self.direction.command + str(self.num_steps - 1)

    def reverse(self) -> "PlanPath":
        return PlanPath(self.direction, -self.num_steps)


class PlanRoute:
    def __init__(self, paths: List[PlanPath]):
        self._paths = self.simplify(paths)

    def __repr__(self):
        return self.to_str()

    def __add__(self, other: "PlanRoute") -> "PlanRoute":
        return PlanRoute(self.paths + other.paths)

    def __bool__(self):
        return bool(self._paths)

    @property
    def paths(self):
        return self._paths

    @property
    def num_steps(self):
        return sum(x.num_steps for x in self._paths)

    @classmethod
    def simplify(cls, paths: List[PlanPath]):
        if not paths:
            return paths

        new_paths = []
        last_path = None
        for p in paths:
            if last_path and p.direction == last_path.direction:
                new_paths[-1] = PlanPath(p.direction, p.num_steps + last_path.num_steps)
            else:
                last_path = p
                new_paths.append(p)
        return new_paths

    def command_length(self):
        return len(self.to_str())

    def min_fleet_size(self):
        return min_ship_count_for_flight_plan_len(self.command_length())

    def reverse(self) -> "PlanRoute":
        return PlanRoute([x.reverse() for x in self.paths])

    @property
    def actions(self):
        actions = []
        for p in self.paths:
            actions += [p.direction for _ in range(p.num_steps)]
        return actions

    @classmethod
    def from_str(cls, str_plan: str, current_direction: Action) -> "PlanRoute":
        if current_direction not in ALL_DIRECTIONS:
            raise ValueError(f"Unknown direction `{current_direction}`")

        if not str_plan:
            return PlanRoute([PlanPath(current_direction, np.inf)])

        commands = []
        for x in str_plan:
            if x in COMMAND_TO_ACTION:
                commands.append([])
                commands[-1].append(x)
            elif x.isdigit():
                if not commands:
                    commands = [[]]
                commands[-1].append(x)
            else:
                raise ValueError(f"Unknown command `{x}`.")

        paths = []
        for i, p in enumerate(commands):
            if i == 0 and p[0].isdigit():
                action = current_direction
                num_steps = int("".join(p))
                if num_steps == 0:
                    continue
            else:
                action = COMMAND_TO_ACTION[p[0]]
                if len(p) == 1:
                    num_steps = 1
                else:
                    num_steps = int("".join(p[1:])) + 1

            paths.append(PlanPath(direction=action, num_steps=num_steps))
            if action == Convert:
                break

        if not paths:
            return PlanRoute([PlanPath(current_direction, np.inf)])

        last_direction = paths[-1].direction
        if last_direction != Convert:
            paths[-1] = PlanPath(direction=last_direction, num_steps=np.inf)

        return PlanRoute(paths)

    def to_str(self) -> str:
        s = ""
        for a in self.paths[:-1]:
            s += a.to_str()
        s += self.paths[-1].direction.command
        return s

#%%writefile board.py


# <--->
'''
from basic import (
    Obj,
    collection_rate_for_ship_count,
    max_ships_to_spawn,
    cached_property,
    create_spawn_ships_command,
    create_launch_fleet_command,
)
from geometry import (
    Field,
    Action,
    Point,
    North,
    South,
    Convert,
    PlanPath,
    PlanRoute,
    GAME_ID_TO_ACTION,
)
'''

class _ShipyardAction:
    def to_str(self):
        raise NotImplementedError

    def __repr__(self):
        return self.to_str()


class Spawn(_ShipyardAction):
    def __init__(self, ship_count: int):
        self.ship_count = ship_count

    def to_str(self):
        return create_spawn_ships_command(self.ship_count)


class Launch(_ShipyardAction):
    def __init__(self, ship_count: int, route: "BoardRoute"):
        self.ship_count = ship_count
        self.route = route

    def to_str(self):
        return create_launch_fleet_command(self.ship_count, self.route.plan.to_str())


class DoNothing(_ShipyardAction):
    def __repr__(self):
        return "Do nothing"

    def to_str(self):
        raise NotImplementedError


class BoardPath:
    max_length = 32

    def __init__(self, start: "Point", plan: PlanPath):
        assert plan.num_steps > 0 or plan.direction == Convert

        self._plan = plan

        field = start.field
        x, y = start.x, start.y
        if np.isfinite(plan.num_steps):
            n = plan.num_steps + 1
        else:
            n = self.max_length
        action = plan.direction

        if plan.direction == Convert:
            self._track = []
            self._start = start
            self._end = start
            self._build_shipyard = True
            return

        if action in (North, South):
            track = field.get_column(x, start=y, size=n * action.dy)
        else:
            track = field.get_row(y, start=x, size=n * action.dx)

        self._track = track[1:]
        self._start = start
        self._end = track[-1]
        self._build_shipyard = False

    def __repr__(self):
        start, end = self.start, self.end
        return f"({start.x}, {start.y}) -> ({end.x}, {end.y})"

    def __len__(self):
        return len(self._track)

    @property
    def plan(self):
        return self._plan

    @property
    def points(self):
        return self._track

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end


class BoardRoute:
    def __init__(self, start: "Point", plan: "PlanRoute"):
        paths = []
        for p in plan.paths:
            path = BoardPath(start, p)
            start = path.end
            paths.append(path)

        self._plan = plan
        self._paths = paths
        self._start = paths[0].start
        self._end = paths[-1].end

    def __repr__(self):
        points = []
        for p in self._paths:
            points.append(p.start)
        points.append(self.end)
        return " -> ".join([f"({p.x}, {p.y})" for p in points])

    def __iter__(self) -> Generator["Point", None, None]:
        for p in self._paths:
            yield from p.points

    def __len__(self):
        return sum(len(x) for x in self._paths)

    def points(self) -> List["Point"]:
        points = []
        for p in self._paths:
            points += p.points
        return points

    @property
    def plan(self) -> PlanRoute:
        return self._plan

    def command(self) -> str:
        return self.plan.to_str()

    @property
    def paths(self) -> List[BoardPath]:
        return self._paths

    @property
    def start(self) -> "Point":
        return self._start

    @property
    def end(self) -> "Point":
        return self._end

    def command_length(self) -> int:
        return len(self.command())

    def last_action(self):
        return self.paths[-1].plan.direction

    def expected_kore(self, board: "Board", ship_count: int):
        rate = collection_rate_for_ship_count(ship_count)
        if rate <= 0:
            return 0

        point_to_time = {}
        point_to_kore = {}
        for t, p in enumerate(self):
            point_to_time[p] = t
            point_to_kore[p] = p.kore

        for f in board.fleets:
            for t, p in enumerate(f.route):
                if p in point_to_time and t < point_to_time[p]:
                    point_to_kore[p] *= f.collection_rate

        return sum([kore * rate for kore in point_to_kore.values()])


class PositionObj(Obj):
    def __init__(self, *args, point: Point, player_id: int, board: "Board", **kwargs):
        super().__init__(*args, **kwargs)
        self._point = point
        self._player_id = player_id
        self._board = board

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._game_id}, position={self._point}, player={self._player_id})"

    def dirs_to(self, obj: Union["PositionObj", Point]):
        if isinstance(obj, Point):
            return self._point.dirs_to(obj)
        return self._point.dirs_to(obj.point)

    def distance_from(self, obj: Union["PositionObj", Point]) -> int:
        if isinstance(obj, Point):
            return self._point.distance_from(obj)
        return self._point.distance_from(obj.point)

    @property
    def board(self) -> "Board":
        return self._board

    @property
    def point(self) -> Point:
        return self._point

    @property
    def player_id(self):
        return self._player_id

    @property
    def player(self) -> "Player":
        return self.board.get_player(self.player_id)


class Shipyard(PositionObj):
    def __init__(self, *args, ship_count: int, turns_controlled: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._ship_count = ship_count
        self._turns_controlled = turns_controlled
        self._guard_ship_count = 0
        self.action: Optional[_ShipyardAction] = None

    @property
    def turns_controlled(self):
        return self._turns_controlled

    @property
    def max_ships_to_spawn(self) -> int:
        return max_ships_to_spawn(self._turns_controlled)

    @property
    def ship_count(self):
        return self._ship_count

    @property
    def available_ship_count(self):
        return self._ship_count - self._guard_ship_count

    @property
    def guard_ship_count(self):
        return self._guard_ship_count

    def set_guard_ship_count(self, ship_count):
        assert ship_count <= self._ship_count
        self._guard_ship_count = ship_count

    # shipyard에게 오는 아군 fleet을 파악
    @cached_property
    def incoming_allied_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id == self.player_id and f.route.end == self.point:
                fleets.append(f)
        return fleets

    # shipyard에게 오는 적 fleet을 파악
    @cached_property
    def incoming_hostile_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id != self.player_id and f.route.end == self.point:
                fleets.append(f)
        return fleets


class Fleet(PositionObj):
    def __init__(
        self,
        *args,
        ship_count: int,
        kore: int,
        route: BoardRoute,
        direction: Action,
        **kwargs,
    ):
        assert ship_count > 0
        assert kore >= 0

        super().__init__(*args, **kwargs)

        self._ship_count = ship_count
        self._kore = kore
        self._direction = direction
        self._route = route

    def __gt__(self, other):
        if self.ship_count != other.ship_count:
            return self.ship_count > other.ship_count
        if self.kore != other.kore:
            return self.kore > other.kore
        return self.direction.game_id > other.direction.game_id

    def __lt__(self, other):
        return other.__gt__(self)

    @property
    def ship_count(self):
        return self._ship_count

    @property
    def kore(self):
        return self._kore

    @property
    def route(self):
        return self._route

    @property
    def eta(self):
        return len(self._route)

    def set_route(self, route: BoardRoute):
        self._route = route

    @property
    def direction(self):
        return self._direction

    @property
    def collection_rate(self) -> float:
        return collection_rate_for_ship_count(self._ship_count)

    def expected_kore(self):
        return self._kore + self._route.expected_kore(self._board, self._ship_count)

    def cost(self):
        return self.board.spawn_cost * self.ship_count

    def value(self):
        return self.kore / self.cost()

    def expected_value(self):
        return self.expected_kore() / self.cost()


class FleetPointer:
    def __init__(self, fleet: Fleet):
        self.obj = fleet
        self.point = fleet.point
        self.is_active = True
        self._paths = []
        self._points = self.points()

    def points(self):
        for path in self.obj.route.paths:
            self._paths.append([path.plan.direction, 0])
            for point in path.points:
                self._paths[-1][1] += 1
                yield point

    def update(self):
        if not self.is_active:
            self.point = None
            return
        try:
            self.point = next(self._points)
        except StopIteration:
            self.point = None
            self.is_active = False

    def current_route(self):
        plan = PlanRoute([PlanPath(d, n) for d, n in self._paths])
        return BoardRoute(self.obj.point, plan)


class Player(Obj):
    def __init__(self, *args, kore: float, board: "Board", **kwargs):
        super().__init__(*args, **kwargs)
        self._kore = kore
        self._board = board

    @property
    def kore(self):
        return self._kore

    def fleet_kore(self):
        return sum(x.kore for x in self.fleets)

    def fleet_expected_kore(self):
        return sum(x.expected_kore() for x in self.fleets)

    def is_active(self):
        return len(self.fleets) > 0 or len(self.shipyards) > 0

    @property
    def board(self):
        return self._board

    def _get_objects(self, name):
        d = []
        for x in self._board.__getattribute__(name):
            if x.player_id == self.game_id:
                d.append(x)
        return d

    @cached_property
    def fleets(self) -> List[Fleet]:
        return self._get_objects("fleets")

    @cached_property
    def shipyards(self) -> List[Shipyard]:
        return self._get_objects("shipyards")

    @cached_property
    def ship_count(self) -> int:
        return sum(x.ship_count for x in itertools.chain(self.fleets, self.shipyards))

    @cached_property
    def opponents(self) -> List["Player"]:
        return [x for x in self.board.players if x != self]

    @cached_property
    def expected_fleets_positions(self) -> Dict[int, Dict[Point, int]]:
        """
        time -> point -> fleet
        """
        time_to_fleet_positions = defaultdict(dict)
        for f in self.fleets:
            for time, point in enumerate(f.route):
                time_to_fleet_positions[time][point] = f
        return time_to_fleet_positions

    @cached_property
    def expected_dmg_positions(self) -> Dict[int, Dict[Point, int]]:
        """
        time -> point -> dmg
        """
        time_to_dmg_positions = defaultdict(dict)
        for f in self.fleets:
            for time, point in enumerate(f.route):
                for adjacent_point in point.adjacent_points:
                    point_to_dmg = time_to_dmg_positions[time]
                    if adjacent_point not in point_to_dmg:
                        point_to_dmg[adjacent_point] = 0
                    point_to_dmg[adjacent_point] += f.ship_count
        return time_to_dmg_positions

    def actions(self):
        if self.available_kore() < 0:
            logger.warning("Negative balance. Some ships will not spawn.")

        shipyard_id_to_action = {}
        for sy in self.shipyards:
            if not sy.action or isinstance(sy.action, DoNothing):
                continue

            shipyard_id_to_action[sy.game_id] = sy.action.to_str()
        return shipyard_id_to_action

    def spawn_ship_count(self):
        return sum(
            x.action.ship_count for x in self.shipyards if isinstance(x.action, Spawn)
        )

    def need_kore_for_spawn(self):
        return self.board.spawn_cost * self.spawn_ship_count()

    def available_kore(self):
        return self._kore - self.need_kore_for_spawn()


_FIELD = None


class Board:
    def __init__(self, obs, conf):
        self._conf = Configuration(conf)
        self._step = obs["step"]

        global _FIELD
        if _FIELD is None or self._step == 0:
            _FIELD = Field(self._conf.size)
        else:
            assert _FIELD.size == self._conf.size

        self._field: Field = _FIELD

        id_to_point = {x.game_id: x for x in self._field}

        for point_id, kore in enumerate(obs["kore"]):
            point = id_to_point[point_id]
            point.set_kore(kore)

        self._players = []
        self._fleets = []
        self._shipyards = []
        for player_id, player_data in enumerate(obs["players"]):
            player_kore, player_shipyards, player_fleets = player_data
            player = Player(game_id=player_id, kore=player_kore, board=self)
            self._players.append(player)

            for fleet_id, fleet_data in player_fleets.items():
                point_id, kore, ship_count, direction, flight_plan = fleet_data
                position = id_to_point[point_id]
                direction = GAME_ID_TO_ACTION[direction]
                if ship_count < self.shipyard_cost and Convert.command in flight_plan:
                    # can't convert
                    flight_plan = "".join(
                        [x for x in flight_plan if x != Convert.command]
                    )
                plan = PlanRoute.from_str(flight_plan, direction)
                route = BoardRoute(position, plan)
                fleet = Fleet(
                    game_id=fleet_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    kore=kore,
                    route=route,
                    direction=direction,
                    board=self,
                )
                self._fleets.append(fleet)

            for shipyard_id, shipyard_data in player_shipyards.items():
                point_id, ship_count, turns_controlled = shipyard_data
                position = id_to_point[point_id]
                shipyard = Shipyard(
                    game_id=shipyard_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    turns_controlled=turns_controlled,
                    board=self,
                )
                self._shipyards.append(shipyard)

        self._players = [x for x in self._players if x.is_active()]

        self._update_fleets_destination()

    def __getitem__(self, item):
        return self._field[item]

    def __iter__(self):
        return self._field.__iter__()

    @property
    def field(self):
        return self._field

    @property
    def size(self):
        return self._field.size

    @property
    def step(self):
        return self._step

    @property
    def steps_left(self):
        return self._conf.episode_steps - self._step - 1

    @property
    def shipyard_cost(self):
        return self._conf.convert_cost

    @property
    def spawn_cost(self):
        return self._conf.spawn_cost

    @property
    def regen_rate(self):
        return self._conf.regen_rate

    @property
    def max_cell_kore(self):
        return self._conf.max_cell_kore

    @property
    def players(self) -> List[Player]:
        return self._players

    @property
    def fleets(self) -> List[Fleet]:
        return self._fleets

    @property
    def shipyards(self) -> List[Shipyard]:
        return self._shipyards

    def get_player(self, game_id) -> Player:
        for p in self._players:
            if p.game_id == game_id:
                return p
        raise KeyError(f"Player `{game_id}` doas not exists.")

    def get_obj_at_point(self, point: Point) -> Optional[Union[Fleet, Shipyard]]:
        for x in itertools.chain(self.fleets, self.shipyards):
            if x.point == point:
                return x

    def _update_fleets_destination(self):
        """
        trying to predict future positions
        very inaccurate
        """

        shipyard_positions = {x.point for x in self.shipyards}

        fleets = [FleetPointer(f) for f in self.fleets]

        while any(x.is_active for x in fleets):
            for f in fleets:
                f.update()

            # fleet to shipyard
            for f in fleets:
                if f.point in shipyard_positions:
                    f.is_active = False

            # allied fleets
            for player in self.players:
                point_to_fleets = defaultdict(list)
                for f in fleets:
                    if f.is_active and f.obj.player_id == player.game_id:
                        point_to_fleets[f.point].append(f)
                for point_fleets in point_to_fleets.values():
                    if len(point_fleets) > 1:
                        for f in sorted(point_fleets, key=lambda x: x.obj)[:-1]:
                            f.is_active = False

            # fleet to fleet
            point_to_fleets = defaultdict(list)
            for f in fleets:
                if f.is_active:
                    point_to_fleets[f.point].append(f)
            for point_fleets in point_to_fleets.values():
                if len(point_fleets) > 1:
                    for f in sorted(point_fleets, key=lambda x: x.obj)[:-1]:
                        f.is_active = False

            # adjacent damage
            point_to_fleet = {}
            for f in fleets:
                if f.is_active:
                    point_to_fleet[f.point] = f

            point_to_dmg = defaultdict(int)
            for point, fleet in point_to_fleet.items():
                for p in point.adjacent_points:
                    if p in point_to_fleet:
                        adjacent_fleet = point_to_fleet[p]
                        if adjacent_fleet.obj.player_id != fleet.obj.player_id:
                            point_to_dmg[p] += fleet.obj.ship_count

            for point, fleet in point_to_fleet.items():
                dmg = point_to_dmg[point]
                if fleet.obj.ship_count <= dmg:
                    fleet.is_active = False

        for f in fleets:
            f.obj.set_route(f.current_route())


##control.py
#%%writefile control.py

# <--->
'''
from geometry import PlanRoute
from board import Player, Launch, Spawn, Fleet, FleetPointer, BoardRoute
from helpers import is_invitable_victory, find_shortcut_routes
from logger import logger
'''
# <--->

# target에 만족하는 적의 shipyard가 있고 direct attack을 실행할 수 있는 아군의 shipyard가 있을 때 실행
# 적의 fleet을 직접 공격할 fleet을 내보냄
# 경로는 적의 fleet과 중간에 마주치지 않도록 조정
def direct_attack(agent: Player, max_distance: int = 10):
    board = agent.board

    max_distance = min(board.steps_left, max_distance)

    targets = []
    for x in agent.opponents:
        for sy in x.shipyards:
            for fleet in sy.incoming_allied_fleets:
                if fleet.expected_value() > 0.5:
                    targets.append(fleet)

    if not targets:
        return

    shipyards = [
        x for x in agent.shipyards if x.available_ship_count > 0 and not x.action
    ]
    if not shipyards:
        return

    point_to_closest_shipyard = {}
    for p in board:
        closest_shipyard = None
        min_distance = board.size
        for sy in agent.shipyards:
            distance = sy.point.distance_from(p)
            if distance < min_distance:
                min_distance = distance
                closest_shipyard = sy
        point_to_closest_shipyard[p] = closest_shipyard.point

    opponent_shipyard_points = {x.point for x in board.shipyards if x.player_id != agent.game_id}
    for t in targets:
        min_ships_to_send = int(t.ship_count * 1.2)  # 공격대는 타겟의 ship 개수보다 1.2배 많게 설정
        attacked = False

        for sy in shipyards:
        #여기서 available ship count는 shipyard에 존재하는 모든 ship의 개수가 아닌 방어를 위해 남겨둔 ship을 제외한 나머지 ship의 개수다
            if sy.action or sy.available_ship_count < min_ships_to_send:  # 이미 행동중인 shipyard 혹은 필요한 ship만큼 shipyard에 없을경우 direct attack실행을 하지 않음
                continue

            num_ships_to_launch = sy.available_ship_count

            for target_time, target_point in enumerate(t.route, 1):  #타겟까지의 거리가 멀면 실행x
                if target_time > max_distance: 
                    continue

                if sy.point.distance_from(target_point) != target_time:
                    continue

                paths = sy.point.dirs_to(target_point)
                random.shuffle(paths)
                plan = PlanRoute(paths)
                destination = point_to_closest_shipyard[target_point]

                paths = target_point.dirs_to(destination)
                random.shuffle(paths)
                plan += PlanRoute(paths)
                if num_ships_to_launch < plan.min_fleet_size(): #plan에 필요한 size보다 ship개수가 적으면 실행x
                    continue

                route = BoardRoute(sy.point, plan)

                if any(x in opponent_shipyard_points for x in route.points()): #route안에 적의 shipyard가 있으면 실행x
                    continue

                if is_intercept_direct_attack_route(route, agent, direct_attack_fleet=t): #루트 안에 적의 fleet이 있을 경우 실행x
                    continue

                logger.info(
                    f"Direct attack {sy.point}->{target_point}, distance={target_time}"
                )
                sy.action = Launch(num_ships_to_launch, route)
                attacked = True
                break

            if attacked:
                break


def is_intercept_direct_attack_route(
    route: BoardRoute, player: Player, direct_attack_fleet: Fleet
):
    board = player.board

    fleets = [FleetPointer(f) for f in board.fleets if f != direct_attack_fleet]

    for point in route.points()[:-1]:
        for fleet in fleets:
            fleet.update()

            if fleet.point is None:
                continue

            if fleet.point == point:
                return True

            if fleet.obj.player_id != player.game_id:
                for p in fleet.point.adjacent_points:
                    if p == point:
                        return True

    return False

# 내 fleet을 희생하더라도 적의 fleet에게 2배 이상의 피해를 줄 수 있을 때 실행
def adjacent_attack(agent: Player, max_distance: int = 10):
    board = agent.board

    max_distance = min(board.steps_left, max_distance)

    targets = _find_adjacent_targets(agent, max_distance)
    if not targets:
        return

    shipyards = [
        x for x in agent.shipyards if x.available_ship_count > 0 and not x.action
    ]
    if not shipyards:
        return

    fleets_to_be_attacked = set()
    for t in sorted(targets, key=lambda x: (-len(x["fleets"]), x["time"])):
        target_point = t["point"]
        target_time = t["time"]
        target_fleets = t["fleets"]
        if any(x in fleets_to_be_attacked for x in target_fleets):
            continue

        for sy in shipyards:
            if sy.action:
                continue

            distance = sy.distance_from(target_point)
            if distance > target_time:
                continue
            min_ship_count = min(x.ship_count for x in target_fleets)
            num_ships_to_send = min(sy.available_ship_count, min_ship_count)

            routes = find_shortcut_routes(
                board,
                sy.point,
                target_point,
                agent,
                num_ships_to_send,
                route_distance=target_time,
            )
            if not routes:
                continue

            route = random.choice(routes)
            logger.info(
                f"Adjacent attack {sy.point}->{target_point}, distance={distance}, target_time={target_time}"
            )
            sy.action = Launch(num_ships_to_send, route)
            for fleet in target_fleets:
                fleets_to_be_attacked.add(fleet)
            break


def _find_adjacent_targets(agent: Player, max_distance: int = 5):
    board = agent.board
    shipyards_points = {x.point for x in board.shipyards}
    fleets = [FleetPointer(f) for f in board.fleets]
    if len(fleets) < 2:
        return []

    time = 0
    targets = []
    while any(x.is_active for x in fleets) and time <= max_distance:
        time += 1

        for f in fleets:
            f.update()

        point_to_fleet = {
            x.point: x.obj
            for x in fleets
            if x.is_active and x.point not in shipyards_points
        }

        for point in board:
            if point in point_to_fleet or point in shipyards_points:
                continue

            adjacent_fleets = [
                point_to_fleet[x] for x in point.adjacent_points if x in point_to_fleet
            ]
            if len(adjacent_fleets) < 2:
                continue

            if any(x.player_id == agent.game_id for x in adjacent_fleets):
                continue

            targets.append({"point": point, "time": time, "fleets": adjacent_fleets})

    return targets


def _need_more_ships(agent: Player, ship_count: int):
    board = agent.board
    if board.steps_left < 10:
        return False
    if ship_count > _max_ships_to_control(agent):
        return False
    if board.steps_left < 50 and is_invitable_victory(agent):
        return False
    return True


def _max_ships_to_control(agent: Player):
    return max(100, 3 * sum(x.ship_count for x in agent.opponents))


def greedy_spawn(agent: Player):
    board = agent.board

    if not _need_more_ships(agent, agent.ship_count):
        return

    ship_count = agent.ship_count
    max_ship_count = _max_ships_to_control(agent)
    for shipyard in agent.shipyards:
        if shipyard.action:
            continue

        if shipyard.ship_count > agent.ship_count * 0.2 / len(agent.shipyards):
            continue

        num_ships_to_spawn = shipyard.max_ships_to_spawn
        if int(agent.available_kore() // board.spawn_cost) >= num_ships_to_spawn:
            shipyard.action = Spawn(num_ships_to_spawn)

        ship_count += num_ships_to_spawn
        if ship_count > max_ship_count:
            return

#fleet을 최대한 많이 생산
def spawn(agent: Player):
    board = agent.board

    if not _need_more_ships(agent, agent.ship_count):
        return

    ship_count = agent.ship_count
    max_ship_count = _max_ships_to_control(agent)
    for shipyard in agent.shipyards:
        if shipyard.action:
            continue
        num_ships_to_spawn = min(
            int(agent.available_kore() // board.spawn_cost),
            shipyard.max_ships_to_spawn,
        )
        if num_ships_to_spawn:
            shipyard.action = Spawn(num_ships_to_spawn)
            ship_count += num_ships_to_spawn
            if ship_count > max_ship_count:
                return



###defence.py
#%%writefile defence.py

# <--->
"""
from board import Spawn, Player, Launch
from helpers import find_shortcut_routes
from logger import logger
"""
# <--->

# 적의 공격이 예정되어 있을 때 shipyard의 방어를 실행
# 앞으로 공격받을 shipyard에 도착할 아군의 fleet과 생산가능한 ship 개수까지 모두 고려한다.
# 만약 공격이 오는 fleet의 ship이 더 많을 경우 주변 shipyard로부터 도움을 요청한다.
def defend_shipyards(agent: Player):
    board = agent.board

    need_help_shipyards = []
    for sy in agent.shipyards:
        if sy.action:
            continue

        incoming_hostile_fleets = sy.incoming_hostile_fleets
        incoming_allied_fleets = sy.incoming_allied_fleets

        if not incoming_hostile_fleets:
            continue

        incoming_hostile_power = sum(x.ship_count for x in incoming_hostile_fleets)
        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)
        incoming_allied_power = sum(
            x.ship_count
            for x in incoming_allied_fleets
            if x.eta < incoming_hostile_time
        )

        ships_needed = incoming_hostile_power - incoming_allied_power
        if sy.ship_count > ships_needed:
            sy.set_guard_ship_count(min(sy.ship_count, int(ships_needed * 1.1)))
            continue

        # spawn as much as possible
        num_ships_to_spawn = min(
            int(agent.available_kore() // board.spawn_cost), sy.max_ships_to_spawn
        )
        if num_ships_to_spawn:
            logger.debug(f"Spawn ships to protect shipyard {sy.point}")
            sy.action = Spawn(num_ships_to_spawn)

        need_help_shipyards.append(sy)

    for sy in need_help_shipyards:
        incoming_hostile_fleets = sy.incoming_hostile_fleets
        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)

        for other_sy in agent.shipyards:
            if other_sy == sy or other_sy.action or not other_sy.available_ship_count:
                continue

            distance = other_sy.distance_from(sy)
            if distance == incoming_hostile_time - 1:
                routes = find_shortcut_routes(
                    board, other_sy.point, sy.point, agent, other_sy.ship_count
                )
                if routes:
                    logger.info(f"Send reinforcements {other_sy.point}->{sy.point}")
                    other_sy.action = Launch(
                        other_sy.available_ship_count, random.choice(routes)
                    )
            elif distance < incoming_hostile_time - 1:
                other_sy.set_guard_ship_count(other_sy.ship_count)
###                
# <--->
"""
from basic import min_ship_count_for_flight_plan_len
from geometry import Point, Convert, PlanRoute, PlanPath
from board import Player, BoardRoute, Launch
"""
# <--->


def expand(player: Player):
    board = player.board
    num_shipyards_to_create = need_more_shipyards(player)
    if not num_shipyards_to_create:
        return

    shipyard_positions = {x.point for x in board.shipyards}

    shipyard_to_point = find_best_position_for_shipyards(player)

    shipyard_count = 0
    for shipyard, target in shipyard_to_point.items():
        if shipyard_count >= num_shipyards_to_create:
            break

        if shipyard.available_ship_count < board.shipyard_cost or shipyard.action:
            continue

        incoming_hostile_fleets = shipyard.incoming_hostile_fleets
        if incoming_hostile_fleets:
            continue

        target_distance = shipyard.distance_from(target)

        routes = []
        for p in board:
            if p in shipyard_positions:
                continue

            distance = shipyard.distance_from(p) + p.distance_from(target)
            if distance > target_distance:
                continue

            plan = PlanRoute(shipyard.dirs_to(p) + p.dirs_to(target))
            route = BoardRoute(shipyard.point, plan)

            if shipyard.available_ship_count < min_ship_count_for_flight_plan_len(
                len(route.plan.to_str()) + 1
            ):
                continue

            route_points = route.points()
            if any(x in shipyard_positions for x in route_points):
                continue

            if not is_safety_route_to_convert(route_points, player):
                continue

            routes.append(route)

        if routes:
            route = random.choice(routes)
            route = BoardRoute(
                shipyard.point, route.plan + PlanRoute([PlanPath(Convert)])
            )
            shipyard.action = Launch(shipyard.available_ship_count, route)
            shipyard_count += 1


def find_best_position_for_shipyards(player: Player):
    board = player.board
    shipyards = board.shipyards

    shipyard_to_scores = defaultdict(list)
    for p in board:
        if p.kore > 50:
            continue

        closed_shipyard = None
        min_distance = board.size
        for shipyard in shipyards:
            distance = shipyard.point.distance_from(p)
            if shipyard.player_id != player.game_id:
                distance -= 1

            if distance < min_distance:
                closed_shipyard = shipyard
                min_distance = distance

        if (
            not closed_shipyard
            or closed_shipyard.player_id != player.game_id
            or min_distance < 3
            or min_distance > 5
        ):
            continue

        nearby_kore = sum(x.kore for x in p.nearby_points(10))
        nearby_shipyards = sum(1 for x in board.shipyards if x.distance_from(p) < 5)
        score = nearby_kore - 1000 * nearby_shipyards - 1000 * min_distance
        shipyard_to_scores[closed_shipyard].append({"score": score, "point": p})

    shipyard_to_point = {}
    for shipyard, scores in shipyard_to_scores.items():
        if scores:
            scores = sorted(scores, key=lambda x: x["score"])
            point = scores[-1]["point"]
            shipyard_to_point[shipyard] = point

    return shipyard_to_point


def need_more_shipyards(player: Player) -> int:
    board = player.board

    if player.ship_count < 100:
        return 0

    fleet_distance = []
    for sy in player.shipyards:
        for f in sy.incoming_allied_fleets:
            fleet_distance.append(len(f.route))

    if not fleet_distance:
        return 0

    mean_fleet_distance = sum(fleet_distance) / len(fleet_distance)

    shipyard_production_capacity = sum(x.max_ships_to_spawn for x in player.shipyards)

    steps_left = board.steps_left
    if steps_left > 100:
        scale = 3
    elif steps_left > 50:
        scale = 4
    elif steps_left > 10:
        scale = 100
    else:
        scale = 1000

    needed = player.kore > scale * shipyard_production_capacity * mean_fleet_distance
    if not needed:
        return 0

    current_shipyard_count = len(player.shipyards)

    op_shipyard_positions = {
        x.point for x in board.shipyards if x.player_id != player.game_id
    }
    expected_shipyard_count = current_shipyard_count + sum(
        1
        for x in player.fleets
        if x.route.last_action() == Convert or x.route.end in op_shipyard_positions
    )

    opponent_shipyard_count = max(len(x.shipyards) for x in player.opponents)
    opponent_ship_count = max(x.ship_count for x in player.opponents)
    if (
        expected_shipyard_count > opponent_shipyard_count
        and player.ship_count < opponent_ship_count
    ):
        return 0

    if current_shipyard_count < 10:
        if expected_shipyard_count > current_shipyard_count:
            return 0
        else:
            return 1

    return max(0, 5 - (expected_shipyard_count - current_shipyard_count))


def is_safety_route_to_convert(route_points: List[Point], player: Player):
    board = player.board

    target_point = route_points[-1]
    target_time = len(route_points)
    for pl in board.players:
        if pl != player:
            for t, positions in pl.expected_fleets_positions.items():
                if t >= target_time and target_point in positions:
                    return False

    shipyard_positions = {x.point for x in board.shipyards}

    for time, point in enumerate(route_points):
        for pl in board.players:
            if point in shipyard_positions:
                return False

            is_enemy = pl != player

            if point in pl.expected_fleets_positions[time]:
                return False

            if is_enemy:
                if point in pl.expected_dmg_positions[time]:
                    return False

    return True


#%%writefile geometry.py

# <--->
"""
from basic import Obj, cached_call, cached_property, min_ship_count_for_flight_plan_len
"""
# <--->


###
#%%writefile helpers.py


# <--->
"""
from geometry import Point
from board import Board, Player, BoardRoute, PlanRoute
"""
# <--->


def is_intercept_route(
    route: BoardRoute, player: Player, safety=True, allow_shipyard_intercept=False
):
    board = player.board

    if not allow_shipyard_intercept:
        shipyard_points = {x.point for x in board.shipyards}
    else:
        shipyard_points = {}

    for time, point in enumerate(route.points()[:-1]):
        if point in shipyard_points:
            return True

        for pl in board.players:
            is_enemy = pl != player

            if point in pl.expected_fleets_positions[time]:
                return True

            if safety and is_enemy:
                if point in pl.expected_dmg_positions[time]:
                    return True

    return False


def find_shortcut_routes(
    board: Board,
    start: Point,
    end: Point,
    player: Player,
    num_ships: int,
    safety: bool = True,
    allow_shipyard_intercept=False,
    route_distance=None
) -> List[BoardRoute]:
    if route_distance is None:
        route_distance = start.distance_from(end)
    routes = []
    for p in board:
        distance = start.distance_from(p) + p.distance_from(end)
        if distance != route_distance:
            continue

        path1 = start.dirs_to(p)
        path2 = p.dirs_to(end)
        random.shuffle(path1)
        random.shuffle(path2)

        plan = PlanRoute(path1 + path2)

        if num_ships < plan.min_fleet_size():
            continue

        route = BoardRoute(start, plan)

        if is_intercept_route(
            route,
            player,
            safety=safety,
            allow_shipyard_intercept=allow_shipyard_intercept,
        ):
            continue

        routes.append(route)

    return routes


def is_invitable_victory(player: Player):
    if not player.opponents:
        return True

    board = player.board
    if board.steps_left > 100:
        return False

    board_kore = sum(x.kore for x in board) * (1 + board.regen_rate) ** board.steps_left

    player_kore = player.kore + player.fleet_expected_kore()
    opponent_kore = max(x.kore + x.fleet_expected_kore() for x in player.opponents)
    return player_kore > opponent_kore + board_kore


  ###
#  %%writefile logger.py
import os
import logging

FILE = "game.log"
IS_KAGGLE = os.path.exists("/kaggle_simulations")
LEVEL = logging.DEBUG if not IS_KAGGLE else logging.INFO
LOGGING_ENABLED = False


class _FileHandler(logging.FileHandler):
    def emit(self, record):
        if not LOGGING_ENABLED:
            return

        if IS_KAGGLE:
            print(self.format(record))
        else:
            super().emit(record)


def init_logger(_logger):
    if not IS_KAGGLE:
        if os.path.exists(FILE):
            os.remove(FILE)

    while _logger.hasHandlers():
        _logger.removeHandler(_logger.handlers[0])

    _logger.setLevel(LEVEL)
    ch = _FileHandler(FILE)
    ch.setLevel(LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H-%M-%S"
    )
    ch.setFormatter(formatter)
    _logger.addHandler(ch)


logger = logging.getLogger()


#%%writefile main.py
# <--->
"""
from board import Board
from logger import logger, init_logger
from offence import capture_shipyards
from defence import defend_shipyards
from expantion import expand
from mining import mine
from control import spawn, greedy_spawn, adjacent_attack, direct_attack
"""
# <--->





#  %%writefile mining.py
import random
import numpy as np
from typing import List
from collections import defaultdict

# <--->
"""
from geometry import PlanRoute
from board import Player, BoardRoute, Launch, Shipyard
from helpers import is_intercept_route
"""
# <--->


def mine(agent: Player):
    board = agent.board
    if not agent.opponents:
        return

    safety = False
    my_ship_count = agent.ship_count
    op_ship_count = max(x.ship_count for x in agent.opponents)
    if my_ship_count < 2 * op_ship_count:
        safety = True

    op_ship_count = []
    for op in agent.opponents:
        for fleet in op.fleets:
            op_ship_count.append(fleet.ship_count)

    if not op_ship_count:
        mean_fleet_size = 0
        max_fleet_size = np.inf
    else:
        mean_fleet_size = np.percentile(op_ship_count, 75)
        max_fleet_size = int(max(op_ship_count) * 1.1)

    point_to_score = estimate_board_risk(agent)

    shipyard_count = len(agent.shipyards)
    if shipyard_count < 10:
        max_distance = 15
    elif shipyard_count < 20:
        max_distance = 12
    else:
        max_distance = 8

    max_distance = min(int(board.steps_left // 2), max_distance)

    for sy in agent.shipyards:
        if sy.action:
            continue

        free_ships = sy.available_ship_count

        if free_ships <= 2:
            continue

        routes = find_shipyard_mining_routes(
            sy, safety=safety, max_distance=max_distance
        )

        route_to_score = {}
        for route in routes:
            route_points = route.points()

            if all(point_to_score[x] > 0 for x in route_points):
                num_ships_to_launch = free_ships
            else:
                if free_ships < mean_fleet_size:
                    continue
                num_ships_to_launch = min(free_ships, max_fleet_size)

            score = route.expected_kore(board, num_ships_to_launch) / len(route)
            route_to_score[route] = score

        if not route_to_score:
            continue

        routes = sorted(route_to_score, key=lambda x: -route_to_score[x])
        for route in routes:
            if all(point_to_score[x] >= 1 for x in route):
                num_ships_to_launch = free_ships
            else:
                num_ships_to_launch = min(free_ships, 199)
            if num_ships_to_launch < route.plan.min_fleet_size():
                continue
            else:
                sy.action = Launch(num_ships_to_launch, route)
                break


def estimate_board_risk(player: Player):
    board = player.board

    shipyard_to_area = defaultdict(list)
    for p in board:
        closest_shipyard = None
        min_distance = board.size
        for sh in board.shipyards:
            distance = sh.point.distance_from(p)
            if distance < min_distance:
                closest_shipyard = sh
                min_distance = distance

        shipyard_to_area[closest_shipyard].append(p)

    point_to_score = {}
    for sy, points in shipyard_to_area.items():
        if sy.player_id == player.game_id:
            for p in points:
                point_to_score[p] = 1
        else:
            for p in points:
                point_to_score[p] = -1

    return point_to_score


def find_shipyard_mining_routes(
    sy: Shipyard, safety=True, max_distance: int = 15
) -> List[BoardRoute]:
    if max_distance < 1:
        return []

    departure = sy.point
    player = sy.player

    destinations = set()
    for shipyard in sy.player.shipyards:
        siege = sum(x.ship_count for x in shipyard.incoming_hostile_fleets)
        if siege >= shipyard.ship_count:
            continue
        destinations.add(shipyard.point)

    if not destinations:
        return []

    routes = []
    for c in sy.point.nearby_points(max_distance):
        if c == departure or c in destinations:
            continue

        paths = departure.dirs_to(c)
        random.shuffle(paths)
        plan = PlanRoute(paths)
        destination = sorted(destinations, key=lambda x: c.distance_from(x))[0]
        if destination == departure:
            plan += plan.reverse()
        else:
            paths = c.dirs_to(destination)
            random.shuffle(paths)
            plan += PlanRoute(paths)

        route = BoardRoute(departure, plan)

        if is_intercept_route(route, player, safety):
            continue

        routes.append(BoardRoute(departure, plan))

    return routes


#  %%writefile offence.py
import random

import numpy as np
from collections import defaultdict

# <--->
"""
from basic import max_ships_to_spawn
from board import Player, Shipyard, Launch
from helpers import find_shortcut_routes
from logger import logger
"""
# <--->


class _ShipyardTarget:
    def __init__(self, shipyard: Shipyard):
        self.shipyard = shipyard
        self.point = shipyard.point
        self.expected_profit = self._estimate_profit()
        self.reinforcement_distance = self._get_reinforcement_distance()
        self._future_ship_count = self._estimate_future_ship_count()
        self.total_incoming_power = self._get_total_incoming_power()

    def __repr__(self):
        return f"Target {self.shipyard}"

    def estimate_shipyard_power(self, time):
        return self._future_ship_count[time]

    def _get_total_incoming_power(self):
        return sum(x.ship_count for x in self.shipyard.incoming_allied_fleets)

    def _get_reinforcement_distance(self):
        incoming_allied_fleets = self.shipyard.incoming_allied_fleets
        if not incoming_allied_fleets:
            return np.inf
        return min(x.eta for x in incoming_allied_fleets)

    def _estimate_profit(self):
        board = self.shipyard.board
        spawn_cost = board.spawn_cost
        profit = sum(
            2 * x.expected_kore() - x.ship_count * spawn_cost
            for x in self.shipyard.incoming_allied_fleets
        )
        profit += spawn_cost * board.shipyard_cost
        return profit

    def _estimate_future_ship_count(self):
        shipyard = self.shipyard
        player = shipyard.player
        board = shipyard.board

        time_to_fleet_kore = defaultdict(int)
        for sh in player.shipyards:
            for f in sh.incoming_allied_fleets:
                time_to_fleet_kore[len(f.route)] += f.expected_kore()

        shipyard_reinforcements = defaultdict(int)
        for f in shipyard.incoming_allied_fleets:
            shipyard_reinforcements[len(f.route)] += f.ship_count

        spawn_cost = board.spawn_cost
        player_kore = player.kore
        ship_count = shipyard.ship_count
        future_ship_count = [ship_count]
        for t in range(1, board.size + 1):
            ship_count += shipyard_reinforcements[t]
            player_kore += time_to_fleet_kore[t]

            can_spawn = max_ships_to_spawn(shipyard.turns_controlled + t)
            spawn_count = min(int(player_kore // spawn_cost), can_spawn)
            player_kore -= spawn_count * spawn_cost
            ship_count += spawn_count
            future_ship_count.append(ship_count)

        return future_ship_count


def capture_shipyards(agent: Player, max_attack_distance=10):
    board = agent.board
    agent_shipyards = [
        x for x in agent.shipyards if x.available_ship_count >= 3 and not x.action
    ]
    if not agent_shipyards:
        return

    targets = []
    for op_sy in board.shipyards:
        if op_sy.player_id == agent.game_id or op_sy.incoming_hostile_fleets:
            continue
        target = _ShipyardTarget(op_sy)
        # if target.expected_profit > 0:
        targets.append(target)

    if not targets:
        return

    for t in targets:
        shipyards = sorted(
            agent_shipyards, key=lambda x: t.point.distance_from(x.point)
        )

        for sy in shipyards:
            if sy.action:
                continue

            distance = sy.point.distance_from(t.point)
            if distance > max_attack_distance:
                continue

            power = t.estimate_shipyard_power(distance)

            if sy.available_ship_count <= power:
                continue

            num_ships_to_launch = min(sy.available_ship_count, int(power * 1.2))

            routes = find_shortcut_routes(
                board,
                sy.point,
                t.point,
                agent,
                num_ships_to_launch,
            )
            if routes:
                route = random.choice(routes)
                logger.info(
                    f"Attack shipyard {sy.point}->{t.point}"
                )
                sy.action = Launch(num_ships_to_launch, route)
                break
              
#Agent wants to be the last function in the file
def agent(obs, conf):
    if obs["step"] == 0:
        init_logger(logger)

    board = Board(obs, conf)
    step = board.step
    my_id = obs["player"]
    remaining_time = obs["remainingOverageTime"]
    logger.info(f"<step_{step + 1}>, remaining_time={remaining_time:.1f}")

    try:
        a = board.get_player(my_id)
    except KeyError:
        return {}

    if not a.opponents:
        return {}
    defend_shipyards(a)
    capture_shipyards(a)
    adjacent_attack(a)
    direct_attack(a)
    expand(a)
    greedy_spawn(a)
    mine(a)
    spawn(a)

    return a.actions()
              
