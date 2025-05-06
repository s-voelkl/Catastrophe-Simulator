import networkx as nx
import numpy as np
from typing import Any, List, Set
from collections.abc import Iterable


# reachable nodes
def reachable_nodes(G: nx.Graph, start):
    """All reachable nodes from a given start node in a graph G.

    Args:
        G (Graph): Input Graph
        start (Node): Node in the Graph

    Returns:
        List: List of all reachable nodes from the start node
    """
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen


def check_if_all_nodes_reachable(G: nx.Graph, n_nodes: int) -> bool:
    """Check if all nodes in the graph are reachable from a given node.
    Args:
        G (Graph): Input Graph
        n_nodes (int): Number of nodes in the graph
    Returns:
        bool: True if all nodes are reachable, False otherwise
    """
    if n_nodes == 0:
        return False

    sample_node = list(G.nodes())[0]  # Get a sample node from the graph
    print(sample_node)
    nodes_reached = reachable_nodes(G, sample_node)
    return len(nodes_reached) == n_nodes


def check_max_nodes_reachable(G: nx.Graph) -> int:
    """Check the maximum number of reachable nodes in the graph.
    Args:
        G (Graph): Input Graph
    Returns:
        int: Maximum number of reachable nodes
    """
    max_reachable_nodes = 0
    for i in list(G.nodes()):
        reachable = reachable_nodes(G, i)
        if len(reachable) > max_reachable_nodes:
            max_reachable_nodes = len(reachable)
    return max_reachable_nodes


# node connectivity
def check_node_connectivity(G: nx.Graph, n_nodes: int, target_vertacies: int) -> bool:
    """Check if the number of vertacies in the graph is equal to the target value of vertacies.
    Args:
        G (Graph): Input Graph
        n_nodes (int): Number of nodes in the graph
        target_vertacies (int): Target number of vertacies
    Returns:
        bool: True if the number of nodes is equal to the target value, False otherwise
    """
    complete_vertacies: int = (n_nodes * (n_nodes - 1)) / 2
    return complete_vertacies == G.size()


# manual graph creations
def generate_er_graph_manually(n_nodes: int, threshold: float):
    """Generate a random Erdos-Renyi graph.
    Args:
        n_nodes (int): Number of nodes in the graph
        threshold (float): Probability of edge creation between nodes
    Returns:
        Graph: NetworkX's generated Erdos-Renyi graph
    """
    # create graph
    G = nx.Graph()

    # create nodes
    for i in range(n_nodes):
        G.add_node(i)

    # create edges as a triangle of all node entries.
    for i in range(n_nodes):
        for k in range(1 + i, n_nodes):
            p = np.random.random()
            if p <= threshold:
                G.add_edge(i, k)

    # note: copy() !
    return G.copy()


def generate_ring_lattice(n_nodes: int, with_edges: bool = True) -> nx.Graph:
    """Generate a ring lattice graph.
    Args:
        n_nodes (int): Number of nodes in the graph
        with_edges (bool): Whether to create edges between nodes
    Returns:
        Graph: NetworkX's generated ring lattice graph
    """
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i)

    if with_edges:
        # create a simple ring
        for i in range(n_nodes):
            G.add_edge(i, (i + 1) % n_nodes)

        # create the connections to the node in +1 distance
        for i in range(n_nodes):
            G.add_edge(i, (i + 2) % n_nodes)

    return G.copy()


def get_nearest_nodes(G: nx.Graph, eval_node: Any, k_neighbors: int) -> list:
    """Get the k nearest nodes from a given node in the graph.
    Args:
        G (Graph): Input Graph
        eval_node (Node): Node in the Graph
        k_neighbors (int): Number of nearest neighbors to return
        distance (int): Distance from the eval_node
    Returns:
        List: List of the k nearest nodes
    """
    dependents: dict = {}
    distances: dict = {}
    stack = [eval_node]

    while stack:
        node = stack.pop()
        if node not in distances:
            try:
                distances[node] = 1 + distances.get(dependents.get(node))
            except:
                distances[node] = 0

            neighbors: Iterable = G.neighbors(node)
            for i in neighbors:
                stack.append(i)
                if i not in dependents.keys():
                    dependents[i] = node

    # sort
    seen_sorted = list(
        {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    )
    # print(seen_sorted)

    try:
        seen_sorted.pop(0)
    except:
        pass
    return seen_sorted[:k_neighbors]


def get_nearest_nodes_without_edges(
    G: nx.Graph, eval_node: Any, k_neighbors: int
) -> list:
    """Get the k nearest nodes from a node of a graph without existing edges.
    Args:
        G (Graph): Input Graph
        eval_node (Node): Node in the Graph
        k_neighbors (int): Number of nearest neighbors to return

    Returns:
        List: List of the k nearest nodes without existing edges
    """
    # base cases
    if k_neighbors <= 0:
        return []
    elif k_neighbors >= len(G.nodes()):
        return list(G.nodes())

    sorted_nodes = list(G.nodes())
    nearest_nodes = []
    starting_node_index = sorted_nodes.index(eval_node)

    # start and end index for the loop of getting the nearest nodes
    start_index = starting_node_index - (k_neighbors // 2)
    end_index = starting_node_index + ((k_neighbors + 1) // 2) + 1

    # Handle special cases where indices go out of bounds
    if start_index < 0:
        start_index += len(sorted_nodes)
    if end_index > len(sorted_nodes):
        end_index -= len(sorted_nodes)
    if start_index > end_index:
        start_index -= len(sorted_nodes)

    # Get the nearest nodes in the sorted list, without including the eval_node itself
    for i in range(start_index, end_index):
        if sorted_nodes[i] == eval_node:
            continue
        nearest_nodes.append(sorted_nodes[i])

    return nearest_nodes


def connect_nodes(G: nx.Graph, start_node: Any, nearest_nodes: List[Any]) -> nx.Graph:
    """Connect the start node to the k nearest nodes in the graph.
    Args:
        G (Graph): Input Graph
        start_node (Node): Node in the Graph
        nearest_nodes (List): List of nearest nodes to connect to
    Returns:
        Graph: Copy of Graph with added edges between start node and the k nearest nodes.
    """

    for node in nearest_nodes:
        # check if the node is already connected to the start node
        if G.has_edge(start_node, node):
            continue
        G.add_edge(node, start_node)

    return G.copy()


def get_node_degree(G: nx.Graph, node: Any) -> int:
    """Get the degree of a node in the graph.
    Args:
        G (Graph): Input Graph
        node (Node): Node in the Graph
    Returns:
        int: Degree of the node
    """

    return G.degree(node)


def watts_strogath_rewire(G: nx.Graph, p_rewire: float) -> nx.Graph:
    """Rewire the edges of the graph using the Watts-Strogatz model.
    Args:
        G (Graph): Input Graph
        p (float): Probability of rewiring an edge
    Returns:
        Graph: Rewired graph
    """

    edges = list(G.edges)
    # print(edges)

    for u, v in edges:
        # determine if rewire should happen. rdm_number >= p_rewire
        random_number = np.random.random()
        if random_number >= p_rewire:
            continue

        # get all nodes as possible edges, without already existing and self loops
        possible_nodes: Set = set(G.nodes)
        # print(possible_nodes)

        # remove u itself
        possible_nodes -= {u}

        # remove existing neighbors of u
        existing_neighbors: Set = set(G.neighbors(u))
        possible_nodes = possible_nodes - existing_neighbors
        # print(possible_nodes)

        # choose a random node from the possible nodes
        random_node = np.random.choice(list(possible_nodes))

        # remove old edge and build new edge
        if G.has_edge(u, random_node) or G.has_edge(random_node, u):
            # print(
            #     "Node:",
            #     u,
            #     "Old edge:",
            #     v,
            #     "New edge:",
            #     random_node,
            #     "Skipped as already existing!",
            # )
            pass
        else:
            G.remove_edge(u, v)
            G.add_edge(u, random_node)
            # print("Node:", u, "Old edge:", v, "New edge:", random_node)

    return G.copy()


def average_node_degree(G: nx.Graph) -> float:
    node_degrees: List[int] = []

    for i in G.nodes:
        node_degrees.append(get_node_degree(G, i))

    avg_node_degree: float = sum(node_degrees) / len(G.nodes)
    return avg_node_degree


# RingLattice = generate_ring_lattice(10)
# nearest_nodes = get_nearest_nodes(RingLattice, 0, 4)

# RingLattice = generate_ring_lattice(10, False)
# print(get_nearest_nodes_without_edges(RingLattice, 0, 0))
# print()
# print(get_nearest_nodes_without_edges(RingLattice, 5, 0))
# print(get_nearest_nodes_without_edges(RingLattice, 5, 1))
# print(get_nearest_nodes_without_edges(RingLattice, 5, 2))
# print(get_nearest_nodes_without_edges(RingLattice, 5, 3))
# print(get_nearest_nodes_without_edges(RingLattice, 5, 4))
# print(get_nearest_nodes_without_edges(RingLattice, 5, 5))
# print()
# print(get_nearest_nodes_without_edges(RingLattice, 9, 4))

# nx.draw_circular(RingLattice, with_labels=True)
# plt.show()
