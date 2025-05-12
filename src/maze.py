import dataclasses
import networkx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Iterator, Literal
from enum import Enum, auto

@dataclasses.dataclass
class Position:
    x: int
    y: int

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y


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
            self.tile_grid = [
                [Tile(x=x, y=y) for x in range(self.width)] for y in range(self.height)
            ]

    # -------------------------------------------------------------------------
    # GETTER / SETTER
    # -------------------------------------------------------------------------
    def get_tile(self, x: int, y: int) -> Tile | None:
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.tile_grid[y][x]
        return None

    def get_all_tiles(self) -> Iterator[Tile]:
        for row in self.tile_grid:
            yield from row

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

    def get_connected_neighbors(self, tile: Tile) -> List[Tile]:
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

    def get_all_neighbors(self, tile: Tile) -> List[Tile]:
        neighbors = []

        for direction in Direction:
            dx, dy = direction.dxdy
            nx, ny = tile.x + dx, tile.y + dy
            neighbor = self.get_tile(nx, ny)

            if neighbor is not None:
                neighbors.append(neighbor)

        return neighbors

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------
    def calculate_wall_density(self) -> float:
        wall_density = 0.0
        for tile in self.get_all_tiles():
            wall_density += tile.wall_densitiy()
        return wall_density / (self.width * self.height)

    def calculate_axial_symmetry(self) -> Tuple[float, float]:
        north_south_score = 0.0

        half_width = self.width // 2
        for y in range(self.height):
            for x in range(half_width):
                tile_left = self.get_tile(x, y)
                tile_right = self.get_tile(self.width - x - 1, y)

                score = 0.0
                if tile_left.north == tile_right.north:
                    score += 0.25
                if tile_left.south == tile_right.south:
                    score += 0.25
                if tile_left.east == tile_right.west:
                    score += 0.25
                if tile_left.west == tile_right.east:
                    score += 0.25

                north_south_score += score

        north_south_symmetry = north_south_score / (half_width * self.height)

        east_west_score = 0.0

        half_height = self.height // 2
        for y in range(half_height):
            for x in range(self.width):
                tile_top = self.get_tile(x, y)
                tile_bottom = self.get_tile(x, self.height - y - 1)

                score = 0.0
                if tile_top.north == tile_bottom.south:
                    score += 0.25
                if tile_top.south == tile_bottom.north:
                    score += 0.25
                if tile_top.east == tile_bottom.east:
                    score += 0.25
                if tile_top.west == tile_bottom.west:
                    score += 0.25

                east_west_score += score

        east_west_symmetry = east_west_score / (self.width * half_height)

        return north_south_symmetry, east_west_symmetry

    # -------------------------------------------------------------------------
    # AUXILIARY
    # -------------------------------------------------------------------------
    def is_tile_empty(self, tile: Tile) -> bool:
        return tile not in (self.save_zones or []) and tile not in (
            self.survivors or []
        )

    # -------------------------------------------------------------------------
    # VISUALISATION
    # -------------------------------------------------------------------------
    def get_tile_lable(
        self, tile: Tile, mode: Literal["coord", "symbol", "full"] = "coord"
    ) -> str:
        coord = f" {tile.x},{tile.y} "
        symbol = "     "
        text = ""
        if not self.is_tile_empty(tile):
            if tile in self.save_zones:
                symbol = "  E  "
                text += "EXIT\n"
            if tile in self.survivors:
                symbol = "  S  "
                text += "SURV\n"

        match mode:
            case "coord":
                return coord
            case "symbol":
                return symbol
            case "full":
                return coord.strip() + "\n " + text
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    def render_maze(self, mode: Literal["coord", "symbol"] = "coord") -> str:
        lines = []

        height = self.height
        width = self.width

        first_line = "╔"
        for x in range(width):
            tile = self.get_tile(x, 0)
            first_line += "═════" if tile.north else "     "
            first_line += "╤" if x < width - 1 else "╗"
        lines.append(first_line)

        for y in range(height):
            content = ""
            for x in range(width):
                tile = self.get_tile(x, y)
                if x == 0:
                    content += "║" if tile.west else " "
                content += self.get_tile_lable(tile, mode=mode)
                if x == width - 1:
                    content += "║" if tile.east else " "
                else:
                    content += "│" if tile.east else " "
            lines.append(content)

            if y < height - 1:
                sep = "╟"
                for x in range(width):
                    tile = self.get_tile(x, y)
                    sep += "─────" if tile.south else "     "
                    sep += "┼" if x < width - 1 else "╢"
                lines.append(sep)

        last_line = "╚"
        for x in range(width):
            tile = self.get_tile(x, height - 1)
            last_line += "═════" if tile.south else "     "
            last_line += "╧" if x < width - 1 else "╝"
        lines.append(last_line)

        if mode == "symbol":
            lines.append("S: Survivor")
            lines.append("E: Exit (Save Zone)")
        return "\n".join(lines)

    def visualize_graph(self) -> None:
        positioning = {}
        labeldict = {}
        for tile in self.get_all_tiles():
            labeldict[tile] = self.get_tile_lable(tile, mode="full")
            positioning[tile] = (tile.x, self.height - tile.y - 1)

        networkx.draw(
            G=self.to_networkx_graph(),
            pos=positioning,
            with_labels=True,
            labels=labeldict,
            node_size=40,
            node_color="lightblue",
            font_size=10,
            font_color="black",
            edge_color="gray",
        )
        plt.show()

    # -------------------------------------------------------------------------
    # TRANSFORMER
    # -------------------------------------------------------------------------
    def to_networkx_graph(self) -> networkx.Graph:
        graph = networkx.Graph()

        for tile in self.get_all_tiles():
            graph.add_node(tile)
            for neighbor in self.get_connected_neighbors(tile):
                if not graph.has_edge(tile, neighbor):
                    graph.add_edge(tile, neighbor)

        return graph

    def from_dict(self, maze_dict: Dict[Tuple[int, int], Dict[str, int]]) -> None:
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
