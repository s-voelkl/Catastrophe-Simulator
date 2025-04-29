from typing import Dict, List, Tuple

# from agentpy import Agent


class Environment:
    maze: Dict[Tuple[int, int], Dict[str, int]]
    survivor_positions: List[Tuple[int, int]]
    start_positions: List[Tuple[int, int]]
    round: int
    steps: int
    # agent: Agent
