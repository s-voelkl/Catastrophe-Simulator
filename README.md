# Simulation and Analysis of Rescue Operations Using Autonomous Robots

In disaster scenarios such as earthquakes, fires, or building collapses, the rapid and efficient rescue of survivors is crucial. Modern technologies like autonomous robots can play a central role, especially when access for human rescuers is difficult or too dangerous.

As part of this project, a system was developed to *simulate and analyze rescue missions within a maze*. The goal is to algorithmically generate various mazes, evaluate their structural properties, and derive insights for the optimal deployment of a rescue robot.

The implementation documented below closely follows the given task description, which defines both the content and the structural framework of this report. The tasks serve as reference points and a guide for the individual chapters.

## Task Description

1. Implement a method for generating a maze.
2. Evaluate each generated maze using the following metrics:
    - The shortest and average path length within the maze
    - Density: Ratio of walls to open paths relative to the total area of the maze
    - The number of exits and the symmetry of the maze
3. For a selected maze, randomly generate several survivors who need to be rescued.
4. Analyze each selected maze:
    - Where would be the best starting point for a rescue robot to begin the rescue sequence so that all survivors can be rescued as quickly as possible? The robot can carry only one person at a time.
    - Where should the robot ideally transport the survivors to?
5. Visually present your results.

6. Design and implement your own extension of the system.

## Implementation

The currently used sourced files are located here:

- **./schema.py** with all used classes and methods for the agent based modeling
- **./simulation_example.ipynb** with the solution for tasks 1-5
- **./batch_run.ipynb** with the extension of the system for task 6 with a batch-run and a full data-analysis

The module **Mesa** was used for agent based modeling with a Model for the Simulation run and the Agent for the robot. For visualization and graph handling, the module **NetworkX** was chosen. The visualization was done by using **pandas** in combination with **seaborn** and **matplotlib**.

The used sources, documentation and presentation can be found in **./documentation**.

## How to run

1. Get the files from this GitHub Repository.
2. Install the requirements of _./requirements_, for example with pip: "pip install -r ./requirements.txt".
3. Run the Jupyter Notebook file _./simulation_example.ipynb_ (for the individual tasks) or _./batch_run.ipynb_ (time consuming!).

## Keep in mind: Potential for Improvement
Currently, the Jupyter Notebooks, the documentation and presentation are only available in **German**, as the course of this project is being held in this language.

Also, there is **room for improvement** through refactoring and enhancing the code. For example, the Open List used in the pathfinding algorithm is currently stored as a dictionary and must be sorted in ascending order by value during each iteration. A more efficient approach would be to use a queue, such as a priority queue implemented via Python’s heapq library.

Additionally, the initial use of a dictionary as the main data structure became a hindrance later in development. A better alternative would have been to structure the maze as a NetworkX graph or as a list of Tile objects, which would have provided a more suitable and flexible foundation.

Many of these improvements have already been prepared in the **./src project folder**, and tests have shown significantly better performance compared to the old structure. However, due to the limited project timeframe, we were unable to fully integrate these enhancements. Pull requests for better implementations are welcome.
