import dataclasses
from typing import Dict, List, Tuple
from enum import Enum, auto


@dataclasses.dataclass
class Position:
    x: int
    y: int


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

    def count_walls(self):
        return self.north + self.east + self.south + self.west

    def count_openings(self):
        return 4 - self.count_walls()

    def get_wall_densitiy(self):
        return self.count_walls() / self.count_openings()


@dataclasses.dataclass
class Maze:
    width: int = dataclasses.field(default=10, init=True)
    height: int = dataclasses.field(default=10, init=True)

    # y is row number [0, height-1]
    # x is col number [0, width-1]
    tile_grid: List[List[Tile]] | None = dataclasses.field(default=None, init=False)
    save_zones: List[Position] | None = dataclasses.field(default=None, init=False)
    survivors: List[Position] | None = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        if self.tile_grid is None:
            col = []
            for i in range(self.width):
                row = []
                col.append(row)
                for j in range(self.height):
                    row.append(Tile(x=i, y=j))
            self.tile_grid = col

    def get_tile(self, x: int, y: int) -> Tile | None:
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.tile_grid[y][x]
        return None

    def set_tile(self, tile: Tile) -> None:
        if 0 <= tile.y < self.height and 0 <= tile.x < self.width:
            self.tile_grid[tile.y][tile.x] = tile

    def set_wall(self, tile: Tile, direction: Direction, wall: int) -> None:
        # Remove: wall = 0; Add: wall = 1
        # Remove/Add wall on this tile
        setattr(tile, direction.name.lower(), wall)

        dx, dy = direction.dxdy
        nx, ny = tile.x + dx, tile.y + dy

        # Remove/Add wall from neighbor if exists and inside maze
        if 0 <= nx < self.width and 0 <= ny < self.height:
            neighbor = self.get_tile(nx, ny)
            setattr(neighbor, direction.opposite.name.lower(), wall)

    def get_neighbors(self, tile: Tile) -> List[Tile]:
        neighbors = []

        if tile.north == 0:
            neighbors.append(self.get_tile(tile.x, tile.y - 1))
        if tile.east == 0:
            neighbors.append(self.get_tile(tile.x + 1, tile.y))
        if tile.south == 0:
            neighbors.append(self.get_tile(tile.x, tile.y + 1))
        if tile.west == 0:
            neighbors.append(self.get_tile(tile.x - 1, tile.y))

        return [tile for tile in neighbors if tile is not None]

    def print(self):
        height = self.height
        width = self.width

        print("╔" + "╤".join(["═════"] * width) + "╗")

        for y in range(height):
            content = "║"
            for x in range(width):
                tile = self.get_tile(x, y)
                content += f" {x},{y} "
                if x == width - 1:
                    content += "║"
                else:
                    content += "│" if tile.east else " "
            print(content)

            if y < height - 1:
                sep = "╟"
                for x in range(width):
                    tile = self.get_tile(x, y)
                    sep += "─────" if tile.south else "     "
                    sep += "┼" if x < width - 1 else "╢"
                print(sep)

        print("╚" + "╧".join(["═════"] * width) + "╝")

    # TODO check if still needed
    # maybe rename to from_dict()
    def transform_dict_to_tiles(
        self,
        maze_dict: Dict[Tuple[int, int], Dict[str, int]],
    ) -> None:
        # transform the dictionary of tiles into a list of tiles

        for pos, walls in maze_dict.items():
            tile = Tile(
                x=pos[0],
                y=pos[1],
                north=walls["N"],
                east=walls["E"],
                south=walls["S"],
                west=walls["W"],
            )
            self.set_tile(tile)
