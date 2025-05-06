import dataclasses
from typing import Dict, List, Tuple


@dataclasses.dataclass
class Position:
    x: int
    y: int


@dataclasses.dataclass
class Tile(Position):
    north: int = dataclasses.field(default=1, init=True)
    east: int = dataclasses.field(default=1, init=True)
    south: int = dataclasses.field(default=1, init=True)
    west: int = dataclasses.field(default=1, init=True)

    #    def to_dict(self) -> Dict:
    #        return {
    #            "N": self.north,
    #            "E": self.east,
    #            "S": self.south,
    #            "W": self.west,
    #        }

    def count_walls(self):
        return self.north + self.east + self.south + self.west


test_maze = {
    (0, 0): {"N": 1, "E": 0, "S": 1, "W": 1},
    (1, 0): {"N": 1, "E": 1, "S": 0, "W": 0},
    (2, 0): {"N": 1, "E": 0, "S": 1, "W": 1},
    (3, 0): {"N": 1, "E": 0, "S": 1, "W": 0},
    (4, 0): {"N": 1, "E": 0, "S": 1, "W": 0},
    (5, 0): {"N": 1, "E": 0, "S": 1, "W": 0},
    (6, 0): {"N": 1, "E": 0, "S": 1, "W": 0},
    (7, 0): {"N": 1, "E": 1, "S": 1, "W": 0},
    (8, 0): {"N": 1, "E": 0, "S": 1, "W": 1},
    (9, 0): {"N": 1, "E": 1, "S": 1, "W": 0},
    (0, 1): {"N": 0, "E": 1, "S": 1, "W": 1},
    (1, 1): {"N": 0, "E": 1, "S": 0, "W": 1},
    (2, 1): {"N": 1, "E": 0, "S": 0, "W": 1},
    (3, 1): {"N": 1, "E": 1, "S": 1, "W": 0},
    (4, 1): {"N": 0, "E": 0, "S": 1, "W": 1},
    (5, 1): {"N": 1, "E": 1, "S": 0, "W": 0},
    (6, 1): {"N": 0, "E": 0, "S": 1, "W": 1},
    (7, 1): {"N": 1, "E": 0, "S": 1, "W": 0},
    (8, 1): {"N": 1, "E": 1, "S": 0, "W": 0},
    (9, 1): {"N": 0, "E": 0, "S": 0, "W": 1},
    (0, 2): {"N": 0, "E": 1, "S": 0, "W": 1},
    (1, 2): {"N": 0, "E": 1, "S": 0, "W": 1},
    (2, 2): {"N": 0, "E": 0, "S": 1, "W": 1},
    (3, 2): {"N": 0, "E": 1, "S": 1, "W": 0},
    (4, 2): {"N": 1, "E": 0, "S": 0, "W": 1},
    (5, 2): {"N": 0, "E": 0, "S": 1, "W": 0},
    (6, 2): {"N": 1, "E": 1, "S": 0, "W": 0},
    (7, 2): {"N": 0, "E": 0, "S": 1, "W": 1},
    (8, 2): {"N": 1, "E": 0, "S": 1, "W": 0},
    (9, 2): {"N": 0, "E": 1, "S": 0, "W": 0},
    (0, 3): {"N": 0, "E": 1, "S": 0, "W": 1},
    (1, 3): {"N": 1, "E": 0, "S": 0, "W": 1},
    (2, 3): {"N": 1, "E": 1, "S": 0, "W": 0},
    (3, 3): {"N": 1, "E": 0, "S": 0, "W": 1},
    (4, 3): {"N": 0, "E": 1, "S": 1, "W": 0},
    (5, 3): {"N": 1, "E": 0, "S": 0, "W": 1},
    (6, 3): {"N": 1, "E": 1, "S": 1, "W": 0},
    (7, 3): {"N": 0, "E": 1, "S": 0, "W": 1},
    (8, 3): {"N": 0, "E": 0, "S": 1, "W": 1},
    (9, 3): {"N": 1, "E": 1, "S": 0, "W": 0},
    (0, 4): {"N": 0, "E": 0, "S": 0, "W": 1},
    (1, 4): {"N": 1, "E": 0, "S": 1, "W": 0},
    (2, 4): {"N": 0, "E": 0, "S": 1, "W": 0},
    (3, 4): {"N": 1, "E": 1, "S": 1, "W": 0},
    (4, 4): {"N": 1, "E": 0, "S": 0, "W": 1},
    (5, 4): {"N": 1, "E": 0, "S": 1, "W": 0},
    (6, 4): {"N": 1, "E": 0, "S": 1, "W": 0},
    (7, 4): {"N": 1, "E": 1, "S": 0, "W": 0},
    (8, 4): {"N": 1, "E": 0, "S": 0, "W": 1},
    (9, 4): {"N": 0, "E": 1, "S": 1, "W": 0},
    (0, 5): {"N": 1, "E": 0, "S": 0, "W": 1},
    (1, 5): {"N": 1, "E": 1, "S": 1, "W": 0},
    (2, 5): {"N": 1, "E": 0, "S": 0, "W": 1},
    (3, 5): {"N": 0, "E": 0, "S": 1, "W": 0},
    (4, 5): {"N": 0, "E": 1, "S": 1, "W": 0},
    (5, 5): {"N": 0, "E": 0, "S": 1, "W": 1},
    (6, 5): {"N": 1, "E": 0, "S": 1, "W": 0},
    (7, 5): {"N": 1, "E": 0, "S": 1, "W": 0},
    (8, 5): {"N": 0, "E": 1, "S": 1, "W": 0},
    (9, 5): {"N": 0, "E": 1, "S": 0, "W": 1},
    (0, 6): {"N": 0, "E": 0, "S": 1, "W": 1},
    (1, 6): {"N": 1, "E": 0, "S": 1, "W": 0},
    (2, 6): {"N": 1, "E": 0, "S": 1, "W": 0},
    (3, 6): {"N": 1, "E": 1, "S": 0, "W": 0},
    (4, 6): {"N": 1, "E": 0, "S": 0, "W": 1},
    (5, 6): {"N": 1, "E": 1, "S": 0, "W": 0},
    (6, 6): {"N": 0, "E": 1, "S": 1, "W": 1},
    (7, 6): {"N": 0, "E": 0, "S": 1, "W": 1},
    (8, 6): {"N": 1, "E": 1, "S": 0, "W": 0},
    (9, 6): {"N": 0, "E": 1, "S": 0, "W": 1},
    (0, 7): {"N": 0, "E": 0, "S": 0, "W": 1},
    (1, 7): {"N": 1, "E": 0, "S": 1, "W": 0},
    (2, 7): {"N": 1, "E": 0, "S": 1, "W": 0},
    (3, 7): {"N": 1, "E": 0, "S": 1, "W": 0},
    (4, 7): {"N": 0, "E": 1, "S": 1, "W": 0},
    (5, 7): {"N": 1, "E": 0, "S": 1, "W": 1},
    (6, 7): {"N": 1, "E": 0, "S": 0, "W": 0},
    (7, 7): {"N": 1, "E": 1, "S": 0, "W": 0},
    (8, 7): {"N": 0, "E": 0, "S": 1, "W": 1},
    (9, 7): {"N": 1, "E": 1, "S": 0, "W": 0},
    (0, 8): {"N": 0, "E": 1, "S": 0, "W": 1},
    (1, 8): {"N": 1, "E": 0, "S": 1, "W": 1},
    (2, 8): {"N": 1, "E": 0, "S": 1, "W": 0},
    (3, 8): {"N": 0, "E": 1, "S": 1, "W": 0},
    (4, 8): {"N": 1, "E": 0, "S": 0, "W": 1},
    (5, 8): {"N": 0, "E": 1, "S": 1, "W": 0},
    (6, 8): {"N": 0, "E": 0, "S": 1, "W": 1},
    (7, 8): {"N": 0, "E": 1, "S": 1, "W": 0},
    (8, 8): {"N": 1, "E": 0, "S": 0, "W": 1},
    (9, 8): {"N": 0, "E": 0, "S": 1, "W": 0},
    (0, 9): {"N": 1, "E": 0, "S": 0, "W": 1},
    (1, 9): {"N": 1, "E": 0, "S": 1, "W": 0},
    (2, 9): {"N": 1, "E": 0, "S": 1, "W": 0},
    (3, 9): {"N": 1, "E": 0, "S": 0, "W": 0},
    (4, 9): {"N": 1, "E": 1, "S": 1, "W": 0},
    (5, 9): {"N": 0, "E": 0, "S": 0, "W": 1},
    (6, 9): {"N": 0, "E": 1, "S": 0, "W": 0},
    (7, 9): {"N": 1, "E": 0, "S": 0, "W": 1},
    (8, 9): {"N": 1, "E": 0, "S": 1, "W": 0},
    (9, 9): {"N": 1, "E": 1, "S": 0, "W": 0},
}


@dataclasses.dataclass
class Maze:
    # y is row number [0, height-1]
    # x is col number [0, width-1]
    tile_grid: List[List[Tile]] | None = dataclasses.field(default=None, init=False)

    save_zones: List[Position] | None = dataclasses.field(default=None, init=False)
    survivors: List[Position] | None = dataclasses.field(default=None, init=False)

    width: int = dataclasses.field(default=10, init=True)
    height: int = dataclasses.field(default=10, init=True)

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


def main():
    m = Maze(height=10, width=10)
    m.transform_dict_to_tiles(test_maze)
    m.print()


if __name__ == "__main__":
    main()
