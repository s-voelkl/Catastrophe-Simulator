from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import pyamaze
import mesa

# alternative: classes for positions and objects in the maze
# class Position(ABC):
#     # abstract class for positions in the maze
#     x: int
#     y: int

#     @abstractmethod
#     def __init__(self, x: int, y: int):
#         self.x = x
#         self.y = y

# class Tile(Position):
#     def __init__(self, x: int, y: int, walls: Dict[str, int]):
#         super().__init__(x, y)
#         # walls: Dict[str, int] = {"N": 0, "E": 0, "S": 0, "W": 0}
#         self.walls = walls

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
    ):
        # Use the seed kwarg to control the random number generator for reproducibility.
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.round = 0
        self.steps = 0
        self._initialize_maze(width, height)
        self._create_save_zones(n_starts)
        self._create_survivors(n_survivors)

        # TODO: create agents

    # MAZE GENERATION (Task 1, 3)
    def _initialize_maze(
        self, width: int, height: int
    ) -> Dict[Tuple[int, int], Dict[str, int]]:
        # TODO: Replace with random dfs maze generation

        m = pyamaze.maze(width, height)
        m.CreateMaze(loopPercent=20, pattern="h")
        self.maze = m.maze_map

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
