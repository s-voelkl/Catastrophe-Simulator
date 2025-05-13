import dataclasses
import networkx
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Iterator, Literal, Set, Any
from base import Position, Direction, Tile
from entity import Survivor


@dataclasses.dataclass
class Maze:
    width: int = dataclasses.field(default=10, init=True)
    height: int = dataclasses.field(default=10, init=True)

    # y is row number [0, height-1]
    # x is col number [0, width-1]
    tile_grid: List[List[Tile]] | None = dataclasses.field(default=None, init=False)
    save_zones: List[Position] | None = dataclasses.field(default=None, init=False)
    survivors: List[Survivor] | None = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        if self.tile_grid is None:
            self.tile_grid = [
                [Tile(x=x, y=y) for x in range(self.width)] for y in range(self.height)
            ]

        if self.save_zones is None:
            self.save_zones = []

        if self.survivors is None:
            self.survivors = []

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

    def get_survivor(self) -> Iterator[Survivor]:
        yield from self.survivors or []

    def get_save_zones(self) -> Iterator[Survivor]:
        yield from self.save_zones or []

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

    def remove_wall(self, tile: Tile, neighbor: Tile) -> None:
        dx = neighbor.x - tile.x
        dy = neighbor.y - tile.y

        for direction in Direction:
            if direction.dxdy == (dx, dy):
                self.set_wall(tile, direction, 0)
                return  # done

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
        # TODO: change into an Iterator similar to get_all_tiles()
        neighbors = []

        for direction in Direction:
            dx, dy = direction.dxdy
            nx, ny = tile.x + dx, tile.y + dy
            neighbor = self.get_tile(nx, ny)

            if neighbor is not None:
                neighbors.append(neighbor)

        return neighbors

    # -------------------------------------------------------------------------
    # GENERATE MAZE
    # -------------------------------------------------------------------------
    def generate_maze_with_random_dfs(self) -> None:
        initial_tile = self.get_tile(x=0, y=0)
        if not initial_tile:
            raise ValueError("Initial tile not found in tiles list.")

        frontier: List[Tile] = [initial_tile]
        visited: Set[Tile] = {initial_tile}

        while frontier:
            # pop a cell as the current cell
            tile: Tile = frontier.pop()

            # if the cell has any neighbors which have not been visited...
            neighbors = self.get_all_neighbors(tile)
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if unvisited_neighbors:
                # push the current cell to the stack
                frontier.append(tile)

                # choose one of the unvisited neighbors at random (use vertical/horizontal preference here?)
                neighbor = random.choice(unvisited_neighbors)

                # and remove the wall between them
                self.remove_wall(tile, neighbor)

                # then mark the neighbor as visited and push it to the stack
                visited.add(neighbor)
                frontier.append(neighbor)

    def generate_save_zones(self, n_save_zones: int) -> None:
        possible_tiles = self.get_empty_tiles(only_border=True)

        for _ in range(n_save_zones):
            if not possible_tiles:
                print("Not enough space for save zones")
                break

            tile = random.choice(list(possible_tiles))

            if tile.y == self.height - 1:
                self.set_wall(tile, Direction.SOUTH, 0)

            if tile.x == self.width - 1:
                self.set_wall(tile, Direction.EAST, 0)

            if tile.y == 0:
                self.set_wall(tile, Direction.NORTH, 0)

            if tile.x == 0:
                self.set_wall(tile, Direction.WEST, 0)

            self.save_zones.append(Position(x=tile.x, y=tile.y))
            possible_tiles.remove(tile)

    def generate_survivors(self, n_survivors: int) -> None:
        possible_tiles = self.get_empty_tiles()

        for _ in range(n_survivors):
            if not possible_tiles:
                print("Not enough space for save zones")
                break

            tile = random.choice(list(possible_tiles))

            self.survivors.append(Survivor(x=tile.x, y=tile.y))
            possible_tiles.remove(tile)

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
        tile_pos = Position(tile.x, tile.y)
        return all(
            tile_pos != Position(sz.x, sz.y) for sz in (self.save_zones or [])
        ) and all(tile_pos != Position(s.x, s.y) for s in (self.survivors or []))

    def get_empty_tiles(self, only_border: bool = False) -> Set[Tile]:
        return {
            tile
            for tile in self.get_all_tiles()
            if self.is_tile_empty(tile)
            and (
                not only_border
                or tile.x == 0
                or tile.x == self.width - 1
                or tile.y == 0
                or tile.y == self.height - 1
            )
        }

    def find_route_with_A_star(self, G: networkx.Graph, start_pos: Position, target_pos: Position) -> List:

        # TODO: replace frontier with priority queue (https://docs.python.org/3/library/heapq.html)
        def sort_dict_by_val_asc(frontier: Dict[Any, int]) -> List[Any]:
            # sort the frontier by f(n) ascending
            return sorted(frontier.items(), key=lambda item: item[1])

        # get start & end node as node in the graph
        start_node: Tile = None
        for node in G.nodes:
            if node.x == start_pos.x and node.y == start_pos.y:
                start_node = node
                break

        end_node: Tile = None
        for node in G.nodes:
            if node.x == target_pos.x and node.y == target_pos.y:
                end_node = node
                break

        if not start_node or not end_node or (start_node == end_node):
            return []

        route: List[Position] = []

        # open list. Tile and the f(n) = g(n) + h(n).
        # g(n): Cost so far
        # h(n): heuristic - manhattan distance to the target_tile
        # Tile -> ( f(n), g(n), h(n) )
        h_start_node: int = start_pos.manhattan_distance(target_pos)
        frontier: Dict[Position, Tuple[int, int, int]] = {
            start_node: (h_start_node + 0, 0, h_start_node)
        }
        visited: Dict[Position, Tuple[int, int, int]] = dict()  # closed list
        parent_pointers: Dict[Position, Position] = dict()  # child -> parent

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
                h_neighbor: int = neighbor.manhattan_distance(end_node)
                f_neighbor: int = g_neighbor + h_neighbor

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

        # reconstructed path -> route
        if not route:
            print("No route found.")
            return []

        return route

    # -------------------------------------------------------------------------
    # VISUALISATION
    # -------------------------------------------------------------------------
    def get_tile_lable(
        self, tile: Tile, mode: Literal["coord", "symbol", "full"] = "coord"
    ) -> str:
        coord = f"{tile.x},{tile.y}"
        symbol = "     "
        text = ""
        if not self.is_tile_empty(tile):
            tile_pos = Position(tile.x, tile.y)
            if tile_pos in self.save_zones:
                symbol = "  E  "
                text += "EXIT\n"
            if tile_pos in self.survivors:
                symbol = "  S  "
                text += "SURV\n"

        match mode:
            case "coord":
                return coord.center(5)
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

    def visualize_graph(self, G: networkx.Graph) -> None:
        positioning = {}
        labeldict = {}
        for tile in self.get_all_tiles():
            labeldict[tile] = self.get_tile_lable(tile, mode="full")
            positioning[tile] = (tile.x, self.height - tile.y - 1)

        networkx.draw(
            G=G,
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
        G = networkx.Graph()

        for tile in self.get_all_tiles():
            G.add_node(Position(x=tile.x, y=tile.y))
            for neighbor in self.get_connected_neighbors(tile):
                if not G.has_edge(Position(x=tile.x, y=tile.y), Position(x=neighbor.x, y=neighbor.y)):
                    G.add_edge(Position(x=tile.x, y=tile.y), Position(x=neighbor.x, y=neighbor.y))

        return G

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
