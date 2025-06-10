from typing import Tuple, Dict, List, Any


def manhattan_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
    # calculate the manhattan distance between two coordinates
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def euklidean_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
    # calculate the Euclidean distance between two coordinates
    return round(
        ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5, 2
    )


def sort_dict_by_val_asc(frontier: Dict[Any, int]) -> List[Any]:
    # sort the frontier by f(n) ascending
    return sorted(frontier.items(), key=lambda item: item[1])
