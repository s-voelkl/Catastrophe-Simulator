from typing import Dict, List, Tuple, Set
from abc import ABC, abstractmethod
import mesa.agent
import mesa.model
import mesa
import random
from helper_functions import *
import csv
import os
import networkx as nx
import matplotlib.pyplot as plt

CSV_VISUALISATION_FILE = "./output/maze_visualisation.csv"
GRAPH_VISUALISATION_FILE = "./output/graph_visualisation.png"


# alternative: classes for positions and objects in the maze
# TODO: bereits eingebaut
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

    # TODO: nicht mehr benötigt
    def get_tile_in_list_by_pos(pos_x: int, pos_y: int, tiles):
        # tiles: List[Tile]
        # get the tile in the list with the same x and y coordinates
        for tile in tiles:
            if tile.x == pos_x and tile.y == pos_y:
                return tile
        return None

    # TODO: nicht mehr benötigt (leicht anders: gibt alle zurück, ungefiltert)
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

    # TODO: bereits eingebaut
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

    # TODO: bereits eingebaut
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

    # TODO: ecke: beide wände weg.
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

    # TODO: bereits eingebaut: ob zwei tiles verbunden sind
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

    # TODO: bereits eingebaut. Dict nicht mehr benötigt
    def transform_tiles_to_dict(tiles: List) -> Dict[Tuple[int, int], Dict[str, int]]:
        # transform the list of tiles into a dictionary with the tile positions as keys and the walls as values
        maze_dict: Dict[Tuple[int, int], Dict[str, int]] = {}
        for tile in tiles:
            maze_dict[(tile.x, tile.y)] = tile.walls
        return maze_dict

    # TODO: bereits eingebaut.
    def transform_dict_to_tiles(
        maze_dict: Dict[Tuple[int, int], Dict[str, int]],
    ) -> List:
        # transform the dictionary of tiles into a list of tiles
        tiles: List[Tile] = []
        for pos, walls in maze_dict.items():
            tile = Tile(pos[0], pos[1], walls)
            tiles.append(tile)
        return tiles

    # TODO: bereits eingebaut.
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

    # TODO: -> helper_functions. Übergabeparam statt maze: G: nx.Graph
    def find_route(
        maze: Dict[Tuple[int, int], Dict[str, int]], start_tile, target_tile
    ) -> List:
        tiles: List[Tile] = Tile.transform_dict_to_tiles(maze)
        G: nx.Graph = Tile.transform_tiles_to_graph(tiles)

        # get start & end node as node in the graph
        start_node: Tile = None
        for node in G.nodes:
            if node.x == start_tile.x and node.y == start_tile.y:
                start_node = node
                break

        end_node: Tile = None
        for node in G.nodes:
            if node.x == target_tile.x and node.y == target_tile.y:
                end_node = node
                break

        if not start_node:
            # print("Start node not found in graph.")
            return []

        if not end_node:
            # print("End node not found in graph.")
            return []

        if start_node == end_node:
            # print("Start and end node are the same.")
            return []

        # resulting route, made by parent_pointers
        route: List[Tile] = []

        # open list. Tile and the f(n) = g(n) + h(n).
        # g(n): Cost so far
        # h(n): heuristic - manhattan distance to the target_tile
        # Tile -> ( f(n), g(n), h(n) )
        h_start_node: int = manhattan_distance(
            (start_tile.x, start_tile.y), (end_node.x, end_node.y)
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

                # returns the list from start to end, without the start tile
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


# TODO: so belassen
class Survivor:
    tile: Tile
    is_rescued: bool

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


# TODO: so belassen
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
    running: bool

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
        self.running = True

    def step(self):
        self.running = True

        if self.model.all_survivors_rescued():
            self.running = False
            return

        # 1. if not transporting survivor: pick up survivor if it not being rescued already
        if self.transported_survivor is None:
            survivor = self.pick_up_survivor()
            if survivor is not None:
                print(
                    f"Agent {self.unique_id} picked up survivor at ({self.tile.x}, {self.tile.y})"
                )
                return

        # 2. place down survivor if existing and on save zone
        if self.transported_survivor is not None:
            for sz in self.model.save_zones:
                if sz.tile.x == self.tile.x and sz.tile.y == self.tile.y:
                    self.place_down_survivor()
                    print(
                        f"Agent {self.unique_id} placed down survivor at ({self.tile.x}, {self.tile.y})"
                    )
                    return

        # if transporting survivor: move to save zone
        if self.transported_survivor is not None:
            self.move_to_save_zone()
            print(
                f"Agent {self.unique_id} transporting survivor. Moved to save zone. ({self.tile.x}, {self.tile.y})"
            )
            return

        # if not transporting survivor: move to survivor
        if self.transported_survivor is None:
            self.move_to_survivor()
            print(
                f"Agent {self.unique_id} not transporting survivor. Moved to next survivor. ({self.tile.x}, {self.tile.y})"
            )
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
            if (
                su.tile.x == self.tile.x
                and su.tile.y == self.tile.y
                and not su.is_rescued
            ):
                survivor = su
                break
        if not survivor:
            # print("No survivor could be found.")
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
            possible_routes[sz] = Tile.find_route(self.model.maze, self.tile, sz.tile)

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
        self.tiles_moved += len(route)
        # print(
        #     f"Agent {self.unique_id} moved to save zone at ({target_save_zone.tile.x}, {target_save_zone.tile.y})"
        # )

        return self.tile

    def move_to_survivor(self) -> Tile:
        possible_routes: Dict[Survivor, List[Tile]] = {}

        # get nearest survivor, that is not Survivor.rescued
        for s in self.model.survivors:
            if s.is_rescued:
                continue

            # if survivor is already being transported by another agent, skip it
            if any(
                s == ts.transported_survivor
                for ts in self.model.agents_by_type[RobotAgent]
            ):
                continue

            # if survivor is on the same position as another agent, skip it
            if any(
                s.tile.x == ts.tile.x and s.tile.y == ts.tile.y
                for ts in self.model.agents_by_type[RobotAgent]
            ):
                continue

            # find route to survivor
            possible_routes[s] = Tile.find_route(self.model.maze, self.tile, s.tile)

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
        self.tiles_moved += len(route)
        # print(
        #     f"Agent {self.unique_id} ({start_tile.x}, {start_tile.y}) moved to survivor at ({target_survivor.tile.x}, {target_survivor.tile.y})"
        # )

        return self.tile

    def change_tile(self, tile: Tile):
        self.tile = tile

        if self.transported_survivor is not None:
            self.transported_survivor.tile = tile


class EnvironmentModel(mesa.Model):
    width: int
    height: int
    maze: Dict[Tuple[int, int], Dict[str, int]]

    survivors: List[Survivor]
    save_zones: List[SaveZone]
    # robot_agents: mesa.agent.AgentSet
    datacollector: mesa.DataCollector
    running: bool
    total_tiles_moved: int
    total_survivors_picked_up: int
    total_survivors_placed_down: int
    initial_pathlengths: List[int]

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
        self.running = True
        self.total_tiles_moved = 0
        self.total_survivors_picked_up = 0
        self.total_survivors_placed_down = 0

        # setup data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Survivors": "survivors",
                "SaveZones": "save_zones",
                "MazeWidth": "width",
                "MazeHeight": "height",
                "AllSurvivorsRescued": self.all_survivors_rescued,
                "InitialPathlengths": "initial_pathlengths",
                "MeanWallDensity": self.get_mean_wall_density,
                "HorizontalSymmetry": self.check_horizontal_symmetry,
                "VeticalSymmetry": self.check_vertical_symmetry,
                "ExitCount": self.get_exit_count,
                "TotalTilesMoved": "total_tiles_moved",
                "TotalSurvivorsPickedUp": "total_survivors_picked_up",
                "TotalSurvivorsPlacedDown": "total_survivors_placed_down",
            },
            agent_reporters={
                "Tile": "tile",
                "TransportedSurvivor": "transported_survivor",
                "TilesMoved": "tiles_moved",
                "SurvivorsPickedUp": "survivors_picked_up",
                "SurvivorsPlacedDown": "survivors_placed_down",
                "StillRunning": "running",
            },
        )

        # start tile for the agents is a save_zone tile
        for i in range(n_robot_agents):
            start_tile: Tile = None
            if self.save_zones:
                start_tile = random.choice(self.save_zones).tile
            else:
                start_tile = Tile.get_tile_in_list_by_pos(
                    0, 0, Tile.transform_dict_to_tiles(self.maze)
                )
            RobotAgent.create_agents(self, 1, start_tile)

        # end -> collect data
        self.initial_pathlengths = self.get_pathlengths_savezones_to_survivors()
        self.datacollector.collect(self)

    # MESA
    def step(self) -> None:
        # metrics
        self.total_tiles_moved = sum(
            [agent.tiles_moved for agent in self.agents_by_type[RobotAgent]]
        )
        self.total_survivors_picked_up = sum(
            [agent.survivors_picked_up for agent in self.agents_by_type[RobotAgent]]
        )
        self.total_survivors_placed_down = sum(
            [agent.survivors_placed_down for agent in self.agents_by_type[RobotAgent]]
        )

        # simulation stop
        if self.all_survivors_rescued():
            print("All survivors rescued. Stopping simulation.")
            self.running = False
            return

        # activate all agents
        print(f"--- Step: {self.steps} ---")
        self.agents.do("step")
        self.datacollector.collect(self)

    # MAZE GENERATION (Task 1, 3)
    def _initialize_maze(
        self, width: int, height: int
    ) -> Dict[Tuple[int, int], Dict[str, int]]:
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
            if any(
                tile.x == sz.tile.x and tile.y == sz.tile.y for sz in self.save_zones
            ):
                continue

            # if tile is already a survivor, skip it
            if any(tile.x == s.tile.x and tile.y == s.tile.y for s in self.survivors):
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

            # if tile is already a save zone, skip it
            if any(
                tile.x == sz.tile.x and tile.y == sz.tile.y for sz in self.save_zones
            ):
                continue

            # if tile is already a survivor, skip it
            if any(tile.x == s.tile.x and tile.y == s.tile.y for s in self.survivors):
                continue

            possible_tiles.add(tile)

        # choose a random tile for each survivor
        for _ in range(n_survivors):
            if len(possible_tiles) == 0:
                print("Not enough space for survivors")
                break

            rdm_tile = random.choice(list(possible_tiles))
            survivor = Survivor(rdm_tile)

            self.survivors.append(survivor)
            possible_tiles.remove(rdm_tile)

        return self.survivors

    def all_survivors_rescued(self) -> bool:
        return all(survivor.is_rescued for survivor in self.survivors)

    # MAZE METRICS (Task 2)
    def get_pathlengths_savezone_to_survivors(self, save_zone: SaveZone) -> List[int]:
        pathlengths: List[int] = []

        for s in self.survivors:
            if s.is_rescued:
                continue

            for ra in self.agents_by_type[RobotAgent]:
                # skip survivors that are already transported by an agent
                if ra.transported_survivor == s:
                    continue

            # find route from save zone to survivor
            route: List[Tile] = Tile.find_route(self.maze, save_zone.tile, s.tile)
            if not route:
                # print(
                #     f"No route found from save zone ({save_zone.tile.x}, {save_zone.tile.y}) "
                #     + f"to survivor ({s.tile.x}, {s.tile.y})"
                # )
                continue

            pathlengths.append(len(route))
        return pathlengths

    def get_pathlengths_savezones_to_survivors(self) -> List[int]:
        pathlengths: List[int] = []

        for sz in self.save_zones:
            save_zone_pathlengths: List[int] = (
                self.get_pathlengths_savezone_to_survivors(sz)
            )
            pathlengths.extend(save_zone_pathlengths)

        return pathlengths

    def get_pathlengths_savezone_to_savezones(self, save_zone: SaveZone) -> List[int]:
        pathlengths: List[int] = []

        for sz in self.save_zones:
            if save_zone == sz:
                continue

            route: List[Tile] = Tile.find_route(self.maze, save_zone.tile, sz.tile)
            if not route:
                continue

            pathlengths.append(len(route))

        return pathlengths

    def get_max_pathlength(pathlengths: List[int]) -> int:
        return max(pathlengths) if pathlengths else 0

    def get_min_pathlength(pathlengths: List[int]) -> int:
        return min(pathlengths) if pathlengths else 0

    def get_mean_pathlength(pathlengths: List[int]) -> float:
        if len(pathlengths) == 0:
            return 0.0
        return sum(pathlengths) / len(pathlengths)

    # Wall density
    def get_mean_wall_density(self) -> float:
        G: nx.Graph = Tile.transform_tiles_to_graph(
            Tile.transform_dict_to_tiles(self.maze)
        )

        n_edges = G.size()
        max_edges: int = 2 * self.width * self.height - self.width - self.height

        wall_density = (max_edges - n_edges) / max_edges if max_edges > 0 else 0
        return wall_density

    # count of exits
    def get_exit_count(self) -> int:
        return len(self.save_zones)

    # symmetry
    def check_horizontal_symmetry(self) -> float:
        # TODO: Implement check for horizontal symmetry
        # idea: cut maze in half, mirror one half and check if it is equal to the other half
        # --> left_half == mirror(right_half)
        return 0

    def check_vertical_symmetry(self) -> float:
        # TODO: Implement check for vertical symmetry
        # just transpose the maze and check horizontal symmetry?
        return 0

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

            # if tile is survivor, add a SURV to it.
            for s in self.survivors:
                if tile.x == s.tile.x and tile.y == s.tile.y:
                    survivors += 1
            if survivors == 1:
                label += "SURV\n"
            elif survivors > 1:
                label += str(survivors) + "xSURVs\n"

            # if tile is save zone, add a EXIT to it.
            for sz in self.save_zones:
                if tile.x == sz.tile.x and tile.y == sz.tile.y:
                    save_zones += 1
            if save_zones == 1:
                label += "EXIT\n"
            elif save_zones > 1:
                label += str(save_zones) + "xEXITs\n"

            # if tile is agent, add a AGENT to it.
            for ag in self.agents_by_type[RobotAgent]:
                if tile.x == ag.tile.x and tile.y == ag.tile.y:
                    agents += 1
            if agents == 1:
                label += "AGENT\n"
            elif agents > 1:
                label += str(agents) + "xAGENTs\n"

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
