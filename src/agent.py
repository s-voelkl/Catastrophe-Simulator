from typing import Dict
from base import Tile
from entity import Survivor
import mesa


class RobotAgent(mesa.Agent):
    # TODO: needs to be reworked and integrated into the new structure
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
