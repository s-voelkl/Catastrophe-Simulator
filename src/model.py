import mesa
from typing import Dict, Tuple, List, Set
from base import Tile
from entity import Survivor
from agent import RobotAgent
import random
import networkx as nx
import matplotlib.pyplot as plt


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

        density = n_edges / max_edges if max_edges > 0 else 0
        return density

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

        plt.show()
        return None
