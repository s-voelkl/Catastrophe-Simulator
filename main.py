from schema import *


def main():
    print("Catastrophe Simulator")
    print("Sebastian Kleber, Simon Völkl, in 2025\n\n")

    # create environment model
    environment: EnvironmentModel = EnvironmentModel(
        width=10, height=10, n_survivors=5, n_agents=1, n_starts=4, seed=None
    )

    # display (saving not implemented yet) the maze data
    environment.save_metrics()

    # display the maze
    environment.visualize_maze()


if __name__ == "__main__":
    main()
