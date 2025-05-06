from typing import Tuple


def transform_coord_for_visualization(
    maze_height: int, x: int, y: int
) -> Tuple[int, int]:
    # transform the coordinates for visualization
    # pyamaze uses (y, x), both starting at 1
    # example:
    # (1,1)     (1,2)     (1,3)
    # (2,1)     (2,2)     (2,3)
    # (3,1)     (3,2)     (3,3)
    return (maze_height - y, x + 1)
