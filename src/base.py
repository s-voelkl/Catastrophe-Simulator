import dataclasses
from enum import Enum, auto


@dataclasses.dataclass
class Position:
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def manhattan_distance(self, other) -> int:
        # calculate the manhattan distance between two coordinates
        return abs(self.x - other.x) + abs(self.y - other.y)


class Direction(Enum):
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()

    @property
    def dxdy(self):
        return {
            Direction.NORTH: (0, -1),
            Direction.SOUTH: (0, 1),
            Direction.EAST: (1, 0),
            Direction.WEST: (-1, 0),
        }[self]

    @property
    def opposite(self):
        return {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST,
        }[self]


@dataclasses.dataclass
class Tile(Position):
    north: int = dataclasses.field(default=1, init=True)
    east: int = dataclasses.field(default=1, init=True)
    south: int = dataclasses.field(default=1, init=True)
    west: int = dataclasses.field(default=1, init=True)

    def __hash__(self):
        return hash((self.x, self.y))

    def count_walls(self) -> int:
        return self.north + self.east + self.south + self.west

    def count_openings(self) -> int:
        return 4 - self.count_walls()

    def wall_densitiy(self) -> float:
        return self.count_walls() / self.count_openings()
