from typing import Dict, List, Tuple, Set
from abc import ABC, abstractmethod
import mesa.agent
import mesa.model
import pyamaze
import mesa
import random
from helper_functions import *
from copy import copy, deepcopy
import csv, os
import networkx as nx
import matplotlib.pyplot as plt

CSV_VISUALISATION_FILE = "./output/maze_visualisation.csv"
GRAPH_VISUALISATION_FILE = "./output/graph_visualisation.png"


# alternative: classes for positions and objects in the maze
class Position(ABC):
    # abstract class for positions in the maze
    # e.g. the top left corner is (0, height -1), the bottom right corner is (width -1, 0)
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
        else:
            print("Tiles are at the same position or not adjacent, cannot add wall.")

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

    def remove_edge_walls(self, maze_width: int, maze_height: int):
        # "N" edge
        if self.y == maze_height - 1:
            self.walls["N"] = 0

        # "E" edge
        if self.x == maze_width - 1:
            self.walls["E"] = 0

        # "S" edge
        if self.y == 0:
            self.walls["S"] = 0

        # "W" edge
        if self.x == 0:
            self.walls["W"] = 0

        return self

    def check_tiles_connection(self, other) -> bool:
        # go through every orientation of the walls of the tiles
        # true, if connected, false if not connected
        # vertical connection: N, S
        if self.x == other.x:
            if self.y < other.y:
                if self.walls["N"] == 0 and other.walls["S"] == 0:
                    return True
            else:
                if self.walls["S"] == 0 and other.walls["N"] == 0:
                    return True

        # horizontal connection: E, W
        if self.y == other.y:
            if self.x < other.x:
                if self.walls["E"] == 0 and other.walls["W"] == 0:
                    return True
            else:
                if self.walls["W"] == 0 and other.walls["E"] == 0:
                    return True
        return False

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

    def transform_tiles_to_graph(tiles: List) -> nx.Graph:
        # transform the list of tiles into a graph
        # positions as nodes, walls with values 0 (no wall) as edges

        G = nx.Graph()

        # add nodes
        for tile in tiles:
            tile: Tile = tile
            if tile not in G.nodes:
                G.add_node(tile)

        # add edges
        for tile in tiles:
            neighbors = tile.get_neighbors(tiles=tiles)
            for neighbor in neighbors:
                if not G.has_edge(tile, neighbor):
                    if tile.check_tiles_connection(neighbor):
                        # add edge if the tiles are connected (no wall between them)
                        G.add_edge(tile, neighbor)

        return G


class Survivor:
    tile: Tile

    def __init__(self, tile: Tile):
        self.tile = tile
        self.is_rescued = False

    def move(self, new_tile: Tile) -> Tile:
        # move the survivor from the current tile to another tile
        self.tile = new_tile
        return self.tile

    def set_rescued(self) -> bool:
        # set the survivor to rescued
        self.is_rescued = True
        return self.is_rescued


class SaveZone:
    tile: Tile

    def __init__(self, tile: Tile):
        self.tile = tile


class RobotAgent(mesa.Agent):
    model: mesa.Model
    # unique_id: int
    tile: Tile
    transported_survivor: Survivor
    tiles_moved: int
    survivors_picked_up: int
    survivors_placed_down: int

    def __init__(self, model, tile: Tile):
        """Create a new agent.

        Args:
            model (Model): The model instance that contains the agent
            tile (Tile): The tile the agent is starting on
        """

        # self.unique_id = len(model.agents)
        # super().__init__(self.unique_id, model)
        super().__init__(model)
        self.tile = tile
        self.transported_survivor = None
        self.tiles_moved = 0
        self.survivors_picked_up = 0
        self.survivors_placed_down = 0

    def step(self):
        # if on save zone
        if (self.tile.x, self.tile.y) in [
            (sz.tile.x, sz.tile.y) for sz in self.model.save_zones
        ]:
            # with survivor: place down
            if self.transported_survivor is not None:
                print("Transporting survivor. Placing survivor down")
                self.place_down_survivor()
                return

            # without survivor: move next
            if self.transported_survivor is None:
                print("Not transporting survivor. Moving to next survivor")
                self.move_to_survivor()
                return

        # if not on save zone
        # if having survivor -> move to save zone
        if self.transported_survivor is not None:
            print("Transporting survivor. Moving")
            self.move_to_save_zone()
            return

        # if not on save zone and no survivor
        if self.transported_survivor is None:
            print("Not transporting survivor. Trying to pick up survivor")
            self.pick_up_survivor()

            if self.transported_survivor is None:
                print("No survivor found")
            return

    def place_down_survivor(self, rescued: bool = True) -> None:
        # update transported survivor properties, place it down
        self.transported_survivor.tile = self.tile
        if rescued:
            self.transported_survivor.set_rescued()

        self.transported_survivor = None
        self.survivors_placed_down += 1

    def pick_up_survivor(self) -> Survivor:
        survivor: Survivor = None

        for su in self.model.survivors:
            if su.tile.x == self.tile.x and su.tile.y == self.tile.y:
                survivor = su
                break
        if not survivor:
            print("No survivor could be found.")
            return None

        self.transported_survivor = survivor
        self.survivors_picked_up += 1
        return Survivor

    def move_to_save_zone(self) -> Tile:
        possible_routes: Dict[SaveZone, List[Tile]] = {}

        # get nearest save zone
        for sz in self.model.save_zones:
            # skip save zones on same tile
            if sz.tile.x == self.tile.x and sz.tile.y == self.tile.y:
                continue

            # find route to save zone
            possible_routes[sz] = self.find_route(sz.tile)

        if not possible_routes:
            print("No possible routes to save zones.")
            return self.tile

        # get fastest route to save_zone: sort dict by len(List[Tile]) ascending, take first element
        sorted_routes = sorted(possible_routes.items(), key=lambda path: len(path[1]))
        # for route in sorted_routes:
        #     print(
        #         f"Route to save zone ({route[0].tile.x}, {route[0].tile.y}) with length {len(route[1])}"
        #     )

        target_save_zone, route = sorted_routes[0]

        if not route:
            print("No route to a save zone possible.")
            return self.tile

        # Move along the route to the save zone
        self.change_tile(route[-1])
        self.tiles_moved += len(route) - 1  # TODO: check correctness
        print(
            f"Agent {self.unique_id} moved to save zone at ({target_save_zone.tile.x}, {target_save_zone.tile.y})"
        )

        return self.tile

    def move_to_survivor(self) -> Tile:
        possible_routes: Dict[Survivor, List[Tile]] = {}

        # get nearest survivor, that is not Survivor.rescued
        for s in self.model.survivors:
            if s.is_rescued:
                continue

            # skip survivors on same tile
            if s.tile.x == self.tile.x and s.tile.y == self.tile.y:
                continue

            # find route to survivor
            possible_routes[s] = self.find_route(s.tile)

        if not possible_routes:
            print("No possible routes to survivors.")
            return self.tile

        # get fastest route to survivor: sort dict by len(List[Tile]) ascending, take first element
        sorted_routes = sorted(possible_routes.items(), key=lambda path: len(path[1]))
        # for route in sorted_routes:
        #     print(
        #         f"Route to survivor ({route[0].tile.x}, {route[0].tile.y}) with length {len(route[1])}"
        #     )

        target_survivor, route = sorted_routes[0]

        if not route:
            print("No route to a survivor possible.")
            return self.tile

        # Move along the route to the survivor
        start_tile: Tile = self.tile
        self.change_tile(route[-1])
        self.tiles_moved += len(route)  # TODO: check correctness
        print(
            f"Agent {self.unique_id} ({start_tile.x}, {start_tile.y}) moved to survivor at ({target_survivor.tile.x}, {target_survivor.tile.y})"
        )

        return self.tile

    def change_tile(self, tile: Tile):
        self.tile = tile

        if self.transported_survivor is not None:
            self.transported_survivor.tile = tile

    def find_route(self, target_tile: Tile) -> List[Tile]:
        tiles: List[Tile] = Tile.transform_dict_to_tiles(self.model.maze)
        G: nx.Graph = Tile.transform_tiles_to_graph(tiles)

        # get start & end node as node in the graph
        start_node: Tile = None
        for node in G.nodes:
            if node.x == self.tile.x and node.y == self.tile.y:
                start_node = node
                break

        end_node: Tile = None
        for node in G.nodes:
            if node.x == target_tile.x and node.y == target_tile.y:
                end_node = node
                break

        if not start_node:
            print("Start node not found in graph.")
            return []

        if not end_node:
            print("End node not found in graph.")
            return []

        if start_node == end_node:
            print("Start and end node are the same.")
            return []

        # resulting route, made by parent_pointers
        route: List[Tile] = []

        # open list. Tile and the f(n) = g(n) + h(n).
        # g(n): Cost so far
        # h(n): heuristic - manhattan distance to the target_tile
        # Tile -> ( f(n), g(n), h(n) )
        h_start_node: int = manhattan_distance(
            (self.tile.x, self.tile.y), (end_node.x, end_node.y)
        )
        frontier: Dict[Tile, Tuple[int, int, int]] = {
            start_node: (h_start_node + 0, 0, h_start_node)
        }
        visited: Dict[Tile, Tuple[int, int, int]] = dict()  # closed list
        parent_pointers: Dict[Tile, Tile] = dict()  # child -> parent

        while frontier:
            # get node with the least f(n) value, add it to visited
            node = sort_dict_by_val_asc(frontier)[0][0]
            node_costs: Tuple[int, int, int] = frontier[node]
            visited[node] = node_costs
            frontier.pop(node)
            # print(f"Visiting node ({node.x},{node.y}) with costs fgh{node_costs}")

            # if goal reached
            if node == end_node:
                # reconstruct the path from the target tile to the start tile
                while node in parent_pointers:
                    route.append(node)
                    node = parent_pointers[node]
                route.reverse()
                return route

            # generate each neighbor of the node
            for neighbor in G.neighbors(node):
                # set parent of neighbor to node

                # get f(neighbor)
                g_neighbor: int = 1 + node_costs[1]  # g(n') = g(n) + 1
                h_neighbor: int = manhattan_distance(
                    (neighbor.x, neighbor.y), (end_node.x, end_node.y)
                )
                f_neighbor: int = g_neighbor + h_neighbor
                # print(
                #     f"- Neighbor ({neighbor.x},{neighbor.y}) with costs fgh({f_neighbor}, {g_neighbor}, {h_neighbor})"
                # )

                # skip worse/equal neighbor values in the frontier
                if neighbor in frontier.keys() and f_neighbor >= frontier[neighbor][0]:
                    continue

                # skip worse/equal neighbor values in the visited
                if neighbor in visited.keys() and f_neighbor >= visited[neighbor][0]:
                    continue

                if neighbor in frontier.keys():
                    frontier.pop(neighbor)
                if neighbor in visited.keys():
                    visited.pop(neighbor)

                frontier[neighbor] = (f_neighbor, g_neighbor, h_neighbor)
                parent_pointers[neighbor] = node
                # print(
                #     f"- Added parent pointer: ({neighbor.x}, {neighbor.y}) -> ({node.x}, {node.y})"
                # )

        # reconstructed path -> route
        if not route:
            print("No route found.")
            return []

        return route


class EnvironmentModel(mesa.Model):

    width: int
    height: int
    maze: Dict[Tuple[int, int], Dict[str, int]]

    survivors: List[Survivor]
    save_zones: List[SaveZone]
    # robot_agents: mesa.agent.AgentSet
    data_collector: mesa.DataCollector

    def __init__(
        self,
        width: int,
        height: int,
        n_survivors: int,
        n_save_zones: int,
        n_robot_agents: int,
        seed=None,
    ):
        # Use the seed kwarg to control the random number generator for reproducibility.
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.survivors = []
        self.save_zones = []
        self.maze = {}
        self._initialize_maze(width, height)
        self._create_save_zones(n_save_zones)
        self._create_survivors(n_survivors)

        # setup data collection
        self.data_collector = mesa.DataCollector(
            model_reporters={
                "Survivors": "survivors",
                "SaveZones": "save_zones",
                "MazeWidth": "width",
                "MazeHeight": "height",
                "AllSurvivorsRescued": self.all_survivors_rescued,
            },
            agent_reporters={
                "Tile": "tile",
                "TransportedSurvivor": "transported_survivor",
                "TilesMoved": "tiles_moved",
                "SurvivorsPickedUp": "survivors_picked_up",
                "SurvivorsPlacedDown": "survivors_placed_down",
            },
        )

        # TODO: create agents
        # start tile for the agents is a save_zone tile
        start_tile: Tile = None
        if self.save_zones:
            start_tile = random.choice(self.save_zones).tile
        else:
            start_tile = Tile.get_tile_in_list_by_pos(
                0, 0, Tile.transform_dict_to_tiles(self.maze)
            )  # TODO: check if this is correct
        RobotAgent.create_agents(self, n_robot_agents, start_tile)

        # end -> collect data
        self.running = True
        self.data_collector.collect(self)

    # MESA
    def step(self) -> None:
        # activate all agents
        self.agents.do("step")
        self.data_collector.collect(self)

    # MAZE GENERATION (Task 1, 3)
    def _initialize_maze(
        self, width: int, height: int
    ) -> Dict[Tuple[int, int], Dict[str, int]]:
        # TODO: Replace with random dfs maze generation (= recursive backtracking)
        # # m = pyamaze.maze(width, height)
        # # m.CreateMaze(loopPercent=20, pattern="h")
        # # self.maze = m.maze_map
        self.maze = {}

        # initialize maze with width*height tiles and all walls present
        tiles: List[Tile] = []
        for h in range(height):
            for w in range(width):
                tiles.append(Tile(w, h))

        initial_tile = Tile.get_tile_in_list_by_pos(0, 0, tiles)
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

                # choose one of the unvisited neighbors at random (use vertical/horizontal preference here?)
                neighbor = random.choice(unvisited_neighbors)

                # and remove the wall between them
                tile.remove_wall(neighbor)

                # then mark the neighbor as visited and push it to the stack
                visited.add(neighbor)
                frontier.append(neighbor)

        # set the maze to the dictionary of tiles
        maze = Tile.transform_tiles_to_dict(tiles)
        self.maze = maze
        return maze

    def _create_save_zones(self, n_save_zones: int) -> None:
        # get all positions
        possible_tiles: Set[Tile] = set()
        tiles: List[Tile] = Tile.transform_dict_to_tiles(self.maze)

        for tile in tiles:
            # tile must be at the borders (top, right, bottom, left)
            if not (
                tile.x == 0
                or tile.x == self.width - 1
                or tile.y == 0
                or tile.y == self.height - 1
            ):
                continue

            # if tile is already a save zone, skip it
            for sz in self.save_zones:
                if tile.x == sz.tile.x and tile.y == sz.tile.y:
                    continue

            # if tile is already a survivor, skip it
            for s in self.survivors:
                if tile.x == s.tile.x and tile.y == s.tile.y:
                    continue

            possible_tiles.add(tile)

        # choose a random tile for each survivor
        for _ in range(n_save_zones):

            if len(possible_tiles) == 0:
                print("Not enough space for save positions")
                break

            tile = random.choice(list(possible_tiles))

            # remove wall between the position and the edge (maze open there)
            tile.remove_edge_walls(self.width, self.height)

            self.save_zones.append(SaveZone(tile))
            possible_tiles.remove(tile)

        return self.save_zones

    def _create_survivors(self, n_survivors: int) -> None:
        # get all positions
        possible_tiles: Set[Tile] = set()
        tiles: List[Tile] = Tile.transform_dict_to_tiles(self.maze)

        for tile in tiles:
            # tile must be accessible (max. 3 walls)
            if 1 not in tile.walls.values():
                continue

            # ignored now: tile cant be at the borders (top, right, bottom, left)
            # if (
            #     tile.x == 0
            #     or tile.x == self.width - 1
            #     or tile.y == 0
            #     or tile.y == self.height - 1
            # ):
            #     continue

            # if tile is already a save zone, skip it
            for sz in self.save_zones:
                if tile.x == sz.tile.x and tile.y == sz.tile.y:
                    continue

            # if tile is already a survivor, skip it
            for s in self.survivors:
                if tile.x == s.tile.x and tile.y == s.tile.y:
                    continue

            possible_tiles.add(tile)

        # choose a random tile for each survivor
        for _ in range(n_survivors):

            if len(possible_tiles) == 0:
                print("Not enough space for survivors")
                break

            tile = random.choice(list(possible_tiles))

            self.survivors.append(Survivor(tile))
            possible_tiles.remove(tile)

        return self.survivors

    def all_survivors_rescued(self) -> bool:
        return all(survivor.is_rescued for survivor in self.survivors)

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
        return len(self.save_zones)

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

    def visualize_graph(self) -> None:
        tiles: List[Tile] = Tile.transform_dict_to_tiles(self.maze)

        G: nx.Graph = Tile.transform_tiles_to_graph(tiles)

        labeldict = {}
        for tile in tiles:
            label: str = ""
            survivors: int = 0
            save_zones: int = 0
            agents: int = 0

            # if tile is survivor, add a "SURV" to it.
            for s in self.survivors:
                if tile.x == s.tile.x and tile.y == s.tile.y:
                    survivors += 1
            if survivors == 1:
                label += "SURV\n"
            elif survivors > 1:
                label += str(survivors) + "SURVs\n"

            # if tile is save zone, add a "SAVEZONE" to it.
            for sz in self.save_zones:
                if tile.x == sz.tile.x and tile.y == sz.tile.y:
                    save_zones += 1
            if save_zones == 1:
                label += "SAVE\n"
            elif save_zones > 1:
                label += str(save_zones) + "SAVEs\n"

            # if tile is agent, add a "AGENT" to it.
            for ag in self.agents_by_type[RobotAgent]:
                if tile.x == ag.tile.x and tile.y == ag.tile.y:
                    agents += 1
            if agents == 1:
                label += "AGENT\n"
            elif agents > 1:
                label += str(agents) + "AGENTs\n"

            label += str(tile.x) + "," + str(tile.y)
            labeldict[tile] = label

        positioning = {}
        for tile in tiles:
            # position the tile in the graph
            positioning[tile] = (tile.x, tile.y)

        nx.draw(
            G,
            pos=positioning,
            with_labels=True,
            labels=labeldict,
            node_size=40,
            node_color="lightblue",
            font_size=10,
            font_color="black",
            edge_color="gray",
        )
        plt.savefig(GRAPH_VISUALISATION_FILE, dpi=300, bbox_inches="tight")
        plt.show()
        return None

    def _save_maze_csv(self) -> str:
        # save the maze to a csv file in the following format:
        #   cell  ,E,W,N,S
        # "(10, 1)",1,1,1,1
        # "(10, 2)",1,1,0,1
        # "(10, 3)",1,1,0,1 ...

        # own structure --> transformed structure for pyamaze
        # {(0, 0):  {'N': 1, 'E': 0, 'S': 1, 'W': 1},   (0, 1): {'N': 0, 'E': 0, 'S': 1, 'W': 1}, (0, 2): {'N': 1, 'E': 0, 'S': 0, 'W': 1}
        # {(10, 1): {'N': 1, 'E': 0, 'S': 1, 'W': 1},   (9, 1): {'N': 0, 'E': 0, 'S': 1, 'W': 1}, (8, 1): {'N': 1, 'E': 0, 'S': 0, 'W': 1}
        transformed_maze: Dict[Tuple[int, int], Dict[str, int]] = {}
        for key, val in self.maze.items():
            # transform the coordinates for visualization
            transformed_key = transform_coord_for_visualization(
                self.height, key[0], key[1]
            )

            # sort the walls in the correct order for pyamaze: E, W, N, S
            transformed_val = {
                "E": val["E"],
                "W": val["W"],
                "N": val["N"],
                "S": val["S"],
            }
            transformed_maze[transformed_key] = transformed_val

        # Transform the transformed maze:
        # Sort the keys by: key[0] descending, key[1] ascending
        transformed_maze = dict(
            sorted(transformed_maze.items(), key=lambda item: (item[0][1], item[0][0]))
        )

        # code from pyamaze to simulate the same csv output as pyamaze
        with open(CSV_VISUALISATION_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["  cell  ", "E", "W", "N", "S"])

            for key, val in transformed_maze.items():
                entry = [key]
                for i in val.values():
                    entry.append(i)
                writer.writerow(entry)

            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 2, os.SEEK_SET)
            f.truncate()

        return CSV_VISUALISATION_FILE

    def visualize_maze(self) -> None:
        m = pyamaze.maze(
            self.height,
            self.width,
        )

        path = self._save_maze_csv()
        m.CreateMaze(loadMaze=path)

        # m.maze_map = deepcopy(transformed_maze)
        # print(m.grid)  # [(1, 1), (2, 1), (3, 1), (4, 1), (5...
        # m.theme = pyamaze.COLOR.dark
        # m.rows = self.height
        # m.cols = self.width
        # m.path = {}
        # m._LabWidth = 26
        # m._goal = (0, 0)

        # manually generate grid for pyamaze
        # grid: List[Tuple[int, int]] = []
        # for w in range(1, self.width + 1):
        #     for h in range(1, self.height + 1):
        #         grid.append((h, w))
        # m.grid = grid
        # m._grid = grid

        # display survivors in the map
        survivor_positions_adjusted = [
            transform_coord_for_visualization(self.height, s.tile.x, s.tile.y)
            for s in self.survivors
        ]

        for survivor_position_adj in survivor_positions_adjusted:
            agent_survivor = pyamaze.agent(
                m,
                # filled=True,
                color=pyamaze.COLOR.red,
                footprints=False,
                x=survivor_position_adj[0],
                y=survivor_position_adj[1],
                shape="square",
            )

        # display save zones in the map
        save_zone_positions_adjusted = [
            transform_coord_for_visualization(self.height, sz.tile.x, sz.tile.y)
            for sz in self.save_zones
        ]
        for save_zone_position_adj in save_zone_positions_adjusted:
            agent_save_zone = pyamaze.agent(
                m,
                # filled=True,
                color=pyamaze.COLOR.green,
                footprints=False,
                x=save_zone_position_adj[0],
                y=save_zone_position_adj[1],
                shape="square",
            )

        m.run()

    def save_metrics(self) -> None:
        # basic metrics
        width = self.width
        height = self.height
        n_survivors = len(self.survivors)
        n_save_zones = len(self.save_zones)

        print("Maze Metrics:")
        print("Width: ", width)
        print("Height: ", height)
        print("Number of Survivors: ", n_survivors)
        print("Number of Save Zones: ", n_save_zones)

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
