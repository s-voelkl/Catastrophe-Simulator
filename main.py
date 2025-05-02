from schema import *


def main():
    print("Catastrophe Simulator")
    print("Sebastian Kleber, Simon Völkl, in 2025\n\n")

    # create environment model
    environment: EnvironmentModel = EnvironmentModel(
        width=10, height=10, n_survivors=5, n_agents=1, n_save_zones=4, seed=None
    )

    print("\nSurvivors:")
    for s in environment.survivor_positions:
        print(f"- Survivor at {s}")

    print("\nSave Zones: ")
    for s in environment.save_zone_positions:
        print(f"- SaveZone at {s}")

    # display (saving not implemented yet) the maze data
    environment.save_metrics()

    # display the maze
    environment.visualize_maze()

    # end
    print("\n\nEnd of simulation")
    print("Sebastian Kleber, Simon Völkl, in 2025")


if __name__ == "__main__":
    main()
