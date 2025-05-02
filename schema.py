from typing import Dict, List, Tuple, Set
from abc import ABC, abstractmethod
import pyamaze
import mesa
import random


# alternative: classes for positions and objects in the maze
class Position(ABC):
    # abstract class for positions in the maze
    x: int
    y: int

    @abstractmethod
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Tile(Position):
    walls: Dict[str, int]
    # walls: N, E, S, W. 0 = no wall, 1 = wall

    def __init__(self, x: int, y: int, walls: Dict[str, int] = None):
        super().__init__(x, y)
        if walls is None:
            walls = {"N": 1, "E": 1, "S": 1, "W": 1}
        self.walls = walls

    def get_tile_in_list_by_pos(pos_x: int, pos_y: int, tiles):
        # tiles: List[Tile]
        # get the tile in the list with the same x and y coordinates
        for tile in tiles:
            if tile.x == pos_x and tile.y == pos_y:
                return tile
        return None

    def get_neighbors(self, tiles):
        # get directly adjacent tiles in the directions N, E, S, W.
        # if a tile does not exist, dont add it to the list of neighbors
        # tiles: List[Tile]
        neighbors: List[Tile] = []
        for direction in ["N", "E", "S", "W"]:
            dx, dy = 0, 0
            if direction == "N":
                dy = 1
            elif direction == "E":
                dx = 1
            elif direction == "S":
                dy = -1
            elif direction == "W":
                dx = -1

            neighbor_x = self.x + dx
            neighbor_y = self.y + dy

            neighbor_tile = Tile.get_tile_in_list_by_pos(neighbor_x, neighbor_y, tiles)
            if neighbor_tile is not None:
                neighbors.append(neighbor_tile)

        return neighbors

    def add_wall(self, neighbor) -> None:
        # add a wall between this tile and the neighbor tile
        if self.x == neighbor.x:
            if self.y < neighbor.y:
                self.walls["N"] = 1
                neighbor.walls["S"] = 1
            else:
                self.walls["S"] = 1
                neighbor.walls["N"] = 1
        elif self.y == neighbor.y:
            if self.x < neighbor.x:
                self.walls["E"] = 1
                neighbor.walls["W"] = 1
            else:
                self.walls["W"] = 1
                neighbor.walls["E"] = 1

    def remove_wall(self, neighbor) -> None:
        # remove a wall between this tile and the neighbor tile
        if self.x == neighbor.x:
            if self.y < neighbor.y:
                self.walls["N"] = 0
                neighbor.walls["S"] = 0
            else:
                self.walls["S"] = 0
                neighbor.walls["N"] = 0
        elif self.y == neighbor.y:
            if self.x < neighbor.x:
                self.walls["E"] = 0
                neighbor.walls["W"] = 0
            else:
                self.walls["W"] = 0
                neighbor.walls["E"] = 0

    def transform_tiles_to_dict(tiles: List) -> Dict[Tuple[int, int], Dict[str, int]]:
        # transform the list of tiles into a dictionary with the tile positions as keys and the walls as values
        maze_dict: Dict[Tuple[int, int], Dict[str, int]] = {}
        for tile in tiles:
            maze_dict[(tile.x, tile.y)] = tile.walls
        return maze_dict

    def transform_dict_to_tiles(
        maze_dict: Dict[Tuple[int, int], Dict[str, int]],
    ) -> List:
        # transform the dictionary of tiles into a list of tiles
        tiles: List[Tile] = []
        for pos, walls in maze_dict.items():
            tile = Tile(pos[0], pos[1], walls)
            tiles.append(tile)
        return tiles


# class Survivor(Position):
#     def __init__(self, x: int, y: int):
#         super().__init__(x, y)
#         self.is_rescued = False

# class SaveZone(Position):
#     def __init__(self, x: int, y: int):
#         super().__init__(x, y)


class EnvironmentModel(mesa.Model):

    width: int
    height: int
    maze: Dict[Tuple[int, int], Dict[str, int]]
    survivor_positions: List[Tuple[int, int]]
    save_zone_positions: List[Tuple[int, int]]
    round: int
    steps: int
    # agent: Agent

    def __init__(
        self,
        width: int,
        height: int,
        n_survivors: int,
        n_starts: int,
        n_agents: int,
        seed=None,
        vertical_priorization: float = 0.5,
    ):
        # Use the seed kwarg to control the random number generator for reproducibility.
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.round = 0
        self.steps = 0
        self._initialize_maze(
            width, height, probability_vertical_neighbor=vertical_priorization
        )
        self._create_save_zones(n_starts)
        self._create_survivors(n_survivors)

        # TODO: create agents

    # MAZE GENERATION (Task 1, 3)
    def _initialize_maze(
        self, width: int, height: int, probability_vertical_neighbor: float = 0.5
    ) -> Dict[Tuple[int, int], Dict[str, int]]:
        self.maze = {}

        # validate inputs
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")

        # if > 0.5: prioritize vertical neighbors --> maze is more vertically aligned
        if not (0 < probability_vertical_neighbor < 1):
            raise ValueError("Probability must be between 0 and 1 (both excluded).")

        # initialize maze with width*height tiles and all walls present
        tiles: List[Tile] = []
        for i in range(height):
            for j in range(width):
                tiles.append(Tile(i, j))

        initial_tile = Tile.get_tile_in_list_by_pos(
            pos_x=0, pos_y=height - 1, tiles=tiles
        )
        if not initial_tile:
            raise ValueError("Initial tile not found in tiles list.")

        frontier: List[Tile] = [initial_tile]
        visited: Set[Tile] = {initial_tile}

        while frontier:
            # pop a cell as the current cell
            tile: Tile = frontier.pop()

            # if the cell has any neighbors which have not been visited...
            neighbors = tile.get_neighbors(tiles)
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if unvisited_neighbors:
                # push the current cell to the stack
                frontier.append(tile)

                # choose one of the unvisited neighbors.
                vertical_neighbors = [
                    n for n in unvisited_neighbors if n.x == tile.x and n.y != tile.y
                ]
                horizontal_neighbors = [
                    n for n in unvisited_neighbors if n.y == tile.y and n.x != tile.x
                ]

                rdm = random.random()
                if vertical_neighbors and horizontal_neighbors:
                    if rdm <= probability_vertical_neighbor:
                        neighbor = random.choice(vertical_neighbors)
                    else:
                        neighbor = random.choice(horizontal_neighbors)
                # if not both, choose one of the two
                elif vertical_neighbors:
                    neighbor = random.choice(vertical_neighbors)
                elif horizontal_neighbors:
                    neighbor = random.choice(horizontal_neighbors)

                # and remove the wall between them
                tile.remove_wall(neighbor)

                # then mark the neighbor as visited and push it to the stack
                visited.add(neighbor)
                frontier.append(neighbor)

        # set the maze to the dictionary of tiles
        maze = Tile.transform_tiles_to_dict(tiles)
        self.maze = maze
        return maze

    def _create_save_zones(self, n_starts: int) -> None:
        # TODO: erstelle Startpositionen im Maze
        self.save_zone_positions = [(1, 1)]
        return

    def _create_survivors(self, n_survivors: int) -> None:
        # TODO: erstelle Überlebende im Maze
        self.survivor_positions = [(3, 3)]
        return

    # MAZE METRICS (Task 2)
    # pathlengths
    def get_pathlengths(self) -> List[int]:
        # TODO: Implement pathlength calculation for the maze
        return [1, 2, 3, 4, 5]

    def get_max_pathlength(self, pathlengths: List[int]) -> int:
        return max(pathlengths) if pathlengths else 0

    def get_min_pathlength(self, pathlengths: List[int]) -> int:
        return min(pathlengths) if pathlengths else 0

    def get_mean_pathlength(self, pathlengths: List[int]) -> float:
        if len(pathlengths) == 0:
            return 0.0
        return sum(pathlengths) / len(pathlengths)

    # Wall density
    def __get_walls_per_tile(self) -> List[int]:
        # TODO: Implement calculation for the wall count for each tile -> List[int]
        return [1, 2, 3, 4, 5]

    def get_mean_wall_density(self) -> int:
        wall_densities: List[int] = self.__get_walls_per_tile()

        if len(wall_densities) == 0:
            return 0

        n_tiles = len(wall_densities)
        density = sum(wall_densities) / n_tiles
        return density

    # count of exits
    def get_exit_count(self) -> int:
        # TODO: Check if this is correct after later data structure changes
        return len(self.save_zone_positions)

    # symmetry
    def check_horizontal_symmetry(self) -> bool:
        # TODO: Implement check for horizontal symmetry
        # idea: cut maze in half, mirror one half and check if it is equal to the other half
        # --> left_half == mirror(right_half)
        return False

    def check_vertical_symmetry(self) -> bool:
        # TODO: Implement check for vertical symmetry
        # just transpose the maze and check horizontal symmetry?
        return False

    # TODO: not needed?
    # def check_point_symmetry(self) -> bool:
    #     pass

    # MAZE VISUALIZATION & OUTPUT (Task 5)
    def visualize_maze(self) -> None:
        m = pyamaze.maze(self.width, self.height)

        # use the maze of self.maze as the grid
        m.grid = self.maze
        m.CreateMaze()
        m.run()

    def save_metrics(self) -> None:
        # basic metrics
        width = self.width
        height = self.height
        n_survivors = len(self.survivor_positions)
        n_save_zones = len(self.save_zone_positions)
        rounds = self.round
        steps = self.steps

        print("Maze Metrics:")
        print("Width: ", width)
        print("Height: ", height)
        print("Number of Survivors: ", n_survivors)
        print("Number of Save Zones: ", n_save_zones)
        print("Rounds: ", rounds)
        print("Steps: ", steps)

        # Pathlengths
        pathlengths = self.get_pathlengths()
        min_pathlength = self.get_min_pathlength(pathlengths)
        mean_pathlength = self.get_mean_pathlength(pathlengths)
        max_pathlength = self.get_max_pathlength(pathlengths)

        print("Min Pathlength: ", min_pathlength)
        print("Mean Pathlength: ", mean_pathlength)
        print("Max Pathlength: ", max_pathlength)

        # Wall densities
        mean_wall_density = self.get_mean_wall_density()
        print("Mean Wall Density: ", mean_wall_density)

        # Exit count
        exit_count = self.get_exit_count()
        print("Exit Count: ", exit_count)

        # Symmetry
        symmetry_horizontal = self.check_horizontal_symmetry()
        symmetry_vertical = self.check_vertical_symmetry()

        print("Horizontal Symmetry: ", symmetry_horizontal)
        print("Vertical Symmetry: ", symmetry_vertical)

        # agent...
        # TODO: write to csv file, use pandas dataframe, ... to save the data?
        # TODO: then visualize the data for n runs with matplotlib/seaborn
