from schema import *


def main():
    print("Catastrophe Simulator")
    print("Sebastian Kleber, Simon Völkl, in 2025\n\n")

    # create environment model
    environment: EnvironmentModel = EnvironmentModel(
        width=10, height=10, n_survivors=5, n_robot_agents=1, n_save_zones=4, seed=None
    )

    print("\nSurvivors:")
    for s in environment.survivors:
        print(f"- Survivor at (x{s.tile.x}, y{s.tile.y}), walls: {s.tile.walls}")

    print("\nSave Zones: ")
    for sz in environment.save_zones:
        print(f"- SaveZone at (x{sz.tile.x}, y{sz.tile.y}), walls: {sz.tile.walls}")

    # display (saving not implemented yet) the maze data
    environment.save_metrics()
    print(environment.maze)

    # display the maze - contains errors when displaying the maze
    # environment.visualize_maze()

    # display the graph
    environment.visualize_graph()

    # end
    print("\n\nEnd of simulation")
    print("Sebastian Kleber, Simon Völkl, in 2025")


if __name__ == "__main__":
    main()
