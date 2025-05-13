import dataclasses
from base import Position


@dataclasses.dataclass
class Survivor(Position):
    is_rescued: bool = dataclasses.field(default=False, init=True)

    def __hash__(self):
        return hash((self.x, self.y))

    # TODO: still needed?
    def move(self, new_pos: Position) -> None:
        self.x = new_pos.x
        self.y = new_pos.y

    def set_rescued(self, value: bool) -> None:
        self.is_rescued = value
