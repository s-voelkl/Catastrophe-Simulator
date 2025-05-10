from typing import Tuple, Dict, List, Any


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


def manhattan_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
    # calculate the manhattan distance between two coordinates
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def sort_dict_by_val_asc(frontier: Dict[Any, int]) -> List[Any]:
    # sort the frontier by f(n) ascending
    return sorted(frontier.items(), key=lambda item: item[1])
