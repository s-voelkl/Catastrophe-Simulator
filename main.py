from schema import *


def main():
    print("Catastrophe Simulator")
    print("Sebastian Kleber, Simon Völkl, in 2025\n\n")

    # create environment model
    environment: EnvironmentModel = EnvironmentModel(
        width=20,
        height=20,
        n_survivors=5,
        n_agents=1,
        n_starts=4,
        seed=None,
        vertical_priorization=0.1,
    )

    # display (saving not implemented yet) the maze data
    environment.save_metrics()

    # display the maze
    environment.visualize_maze()

    # end
    print("\n\nEnd of simulation")
    print("Sebastian Kleber, Simon Völkl, in 2025")


if __name__ == "__main__":
    main()
