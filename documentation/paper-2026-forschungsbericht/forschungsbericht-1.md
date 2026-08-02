# Agent-Based Simulation of Rescue Robots in Maze-Like Disaster Areas – Part 1: Simulation Model and Fundamental Insights

*[Draft for Forschungsbericht 2027, OTH Amberg-Weiden — English text intended for continuous, single-column, 10 pt, non-justified formatting in Word. Citation and image placeholders must be resolved before submission.]*

*[AI disclosure placeholder: Please state here which AI tools were used during research/writing and how, per the OTH AW AI strategy guidance in the submission requirements.]*

---

## Zusammenfassung

Effiziente Rettungseinsätze nach Katastrophen erfordern einen schnellen und sicheren Zugang zu Überleben-den. Roboter-gestützte Mehragentensysteme (MAS) bieten hierfür vielversprechende Möglichkeiten, da sie in gefährlichen Umgebungen eingesetzt werden können und im Vergleich zu menschlichen Einsatzkräften ein geringeres Risiko darstellen. Dieser Bericht stellt eine agentenbasierte Simulation (ABS) vor, in der vierbeini-ge Rettungsroboter labyrinthartige Katastrophengebiete durchqueren, um Überlebende zu sicheren Ret-tungszonen zu bringen. Die Simulation wurde in Python mit der Bibliothek Mesa umgesetzt; die Pfadplanung erfolgt mittels A*-Algorithmus unter Verwendung der Manhattan-Distanz als Heuristik. Labyrinthe werden durch einen iterativen, randomisierten Tiefensuche-Algorithmus (DFS) erzeugt, wodurch realistische, sack-gassenreiche Strukturen entstehen, die Trümmerfeldern ähneln. In einer umfassenden Parameterstudie mit 58.800 Simulationsläufen wurden Labyrinthgröße, Anzahl der Überlebenden, Rettungszonen und Roboter systematisch variiert. Die Ergebnisse zeigen, dass die Anzahl der Simulationsschritte primär von der Anzahl der Überlebenden und Roboter abhängt, während Labyrinthgröße und Zonenanzahl kaum Einfluss darauf haben. Die relative Pfadlänge sinkt mit wachsender Labyrinthfläche, da Sackgassen für die kürzesten Wege irrelevant sind. Die insgesamt zurückgelegte Strecke steigt mit Labyrinthgröße und Überlebendenzahl, sinkt jedoch deutlich mit zunehmender Anzahl an Rettungszonen. Eine höhere Roboteranzahl reduziert zwar die Anzahl der benötigten Simulationsschritte spürbar, führt aber durch fehlende Koordination zu einer leichten Zunahme der zurückgelegten Distanz je Überlebendem. Diese grundlegenden Erkenntnisse bilden die metho-dische und inhaltliche Basis für weitere Arbeiten in der Optimierung der Rettungszonen-Platzierung sowie einer möglichen Realwelt-Umsetzung mit vierbeinigen Robotern.

*(≈ 235 Wörter)*

## Abstract

Disaster response operations require rapid and safe access to survivors to maximize human survivability and minimize casualties. This report presents an agent-based simulation (ABS) study in which quadruped rescue robots navigate grid-like, maze-structured disaster areas to transport survivors to designated safe zones. The simulation environment was implemented in Python using the Mesa framework; robot navigation relies on the A* algorithm with a Manhattan-distance heuristic. Mazes are generated using an iterative randomized depth-first search (DFS) algorithm, producing dead-end-rich structures reminiscent of debris fields in real disaster zones. An exhaustive parameter study comprising 58,800 simulation runs systematically varied maze dimensions, survivor count, safe zone count, and robot count. Results show that the number of simulation steps is determined primarily by the number of survivors and robots, while maze dimensions and safe zone count have negligible influence. Relative path length decreases as maze area grows, since dead-ends become irrelevant for shortest-path computation. The total distance traveled by robots increases with maze size and survivor count but decreases markedly with a higher number of safe zones. Increasing the number of robots noticeably reduces the required simulation steps, yet slightly increases the per-survivor distance traveled due to uncoordinated path interference between agents. These foundational findings establish the methodological and empirical basis for further research in the optimization of safe-zone placement and a possible real-world implementation using quadruped robots.

*(≈ 210 words)*

---

## 1 Introduction

Efficiently rescuing survivors remains a critical challenge in disaster response scenarios. Robotic rescue systems are gaining increasing relevance in this context, as they can be deployed in hazardous environments, communicate with one another, and are more expendable in high-risk operations than human responders.

A multi-agent system (MAS) is described as a collaboration of multiple agents — simple or advanced entities that decide on and perform actions based on the environment and sensed parameters — that jointly solve a complex task, such as a catastrophe rescue operation [#ref Dorri2018Multi].

While the large-scale deployment of MAS for disaster response is not yet feasible [#ref Drew2021Multi], initiatives such as RoboCup Rescue [#ref robocuprescue2025] actively advance the field. Through its Simulation Innovation Strategy, RoboCup Rescue introduces increasingly complex multi-agent challenges, fostering strategic innovation in autonomous coordination and decision-making within simulated disaster scenarios [#ref Visser2015RoboCup].

In a multi-mobile robot system (MMRS) — a specific type of MAS in which all agents are mobile robots [#ref yan2013survey] — resource conflicts frequently arise when multiple robots attempt to access the same physical space or manipulate the same objects. Addressing such conflicts requires effective coordination strategies, such as task and motion planning, to ensure orderly and efficient resource usage [#ref yan2013survey] [#ref Parker2016multiple]. Despite recent advances, simulation-based and real-world MAS implementations still face significant challenges, including learning, fault detection, and localization [#ref Dorri2018Multi].

Navigation and path planning are central components of any MMRS. The A* algorithm [#ref hart1968formal] is a widely used global pathfinding method, offering better performance than Dijkstra's algorithm [#ref Dijkstra1959Note] due to its use of heuristics for search guidance. Our contribution explores the Manhattan-distance heuristic as an intuitive and computationally efficient approximation for spatial measurements within disaster zones.

This report focuses on the systematic evaluation of simulation data generated using agent-based simulation (ABS) [#ref siebers2008introduction] to derive actionable insights concerning maze structure, survivor distribution, and rescue-agent deployment. It introduces the simulation model and presents fundamental findings on maze topology, path lengths, and robot coordination. Future work can build on these results to address the optimal placement of safe zones and discuss a possible real-world implementation using quadruped robots.

The remainder of this report is structured as follows: Section 2 introduces the simulation implementation, Section 3 presents the results of the data analysis, and Section 4 summarizes the findings and outlines future work.

## 2 Methodical Approach

### 2.1 Simulation Environment and Map Generation

Mazes are generated using an iterative variant of randomized depth-first search (DFS), as proposed by Korf [#ref KORF198597], since the resulting structure — rich in dead-ends — closely resembles debris fields encountered in real disaster rescue operations.

Rescue zones, referred to as safe zones (SAFE), are placed randomly along the edges of the maze. These safe zones function as secure locations where survivors receive necessary care and support. Survivors (SURV) are placed randomly within the maze, with the constraint that they may not occupy a safe-zone position. Their locations are known to the agents from the beginning of the simulation, eliminating the need for a dedicated search-and-detection system and thereby substantially reducing the simulation's complexity [#ref GELENBE2012Large].

One or more robot agents (AGENT) are initially deployed at randomly selected safe zones to commence the rescue operation. [#img graph_visualisation.png] shows a small 8×8 example maze with all entity types. Tiles are labeled with their coordinates, and connecting lines indicate possible paths. The movement between two neighboring cells is referred to as the distance moved, representing the cost of traversing from one tile to another.

*(Figure 1: Visualization of the maze map. Survivors (SURV) are transported by robot agents (AGENT) to the rescue zones (SAFE).)*

### 2.2 Agent Behavior and Path Planning

At each simulation step, a robot agent performs one of the following actions: move toward the nearest unclaimed survivor; pick up a survivor; move toward the nearest safe zone; or drop off a survivor. Only survivors who have not yet been rescued are considered when selecting pick-up and drop-off targets.

Robot movement relies on the A* algorithm, implemented using the NetworkX library [#ref hagberg2008exploring]. The algorithm accounts for the path cost already incurred, $g(n)$, and a heuristic estimate of the remaining cost, $h(n)$. The Manhattan distance is used as the heuristic, since it is best suited to grid-based movement restricted to four directions [#ref Patel2025Heuristics]. In each simulation step, robots act sequentially. The objective of the simulation is to move all survivors to designated safe zones in a minimum number of steps and tiles moved.

### 2.3 Experimental Setup

The simulation and batch execution were implemented in Python using the Mesa library [#ref Hoeven2025mesa]. Parameters were varied exhaustively across all combinations of maze width and height (8–20), number of survivors (1–6), number of safe zones (1–6), and number of robot agents (1–6). This factorial design resulted in a total of 58,800 simulation runs, enabling robust statistical evaluation across the parameter space.

The full source code is available at [github.com/s-voelkl/Catastrophe-Simulator](https://github.com/s-voelkl/Catastrophe-Simulator).

## 3 Results

Visualizations in this section use Seaborn's default 95% confidence interval error bands, constructed from bootstrap distributions to reflect uncertainty in the sampled data [#ref Waskom2021seaborn].

### 3.1 Relationships Between Simulation Parameters

A correlation heatmap of the most relevant simulation features, analyzed using Pearson's correlation coefficient [#ref KIRCH2008Pearson], is shown in [#img correlation_heatmap.png]. As displayed, the number of simulation steps is not influenced by the maze's height, width, or number of designated safe zones. Instead, it is determined solely by the number of survivors to be rescued and the number of deployed robots, since robots are not constrained by a maximum movement range per simulation step.

*(Figure 2: Correlation heatmap for the most relevant features of the agent-based simulation.)*

### 3.2 Influence of Maze Structure on Path Lengths

The iterative randomized DFS method frequently produces dead-ends — cells connected to only one neighboring cell. Perfect mazes, characterized by a single entry and exit and the absence of dead-ends, exhibit a maximum path length between two points equal to the total number of tiles in the maze, since only one path connects any two points.

In general, dead-ends increase structural complexity but reduce the effective path length between two arbitrary points, compared to a perfect maze with only one passage. As noted by Bellot et al., a perfect maze with no intersections has a very long solution path, but is "boring and not fun to solve, because the solution is its unique path" [#ref bellot2021generate]. Buck even proposed the number of dead-ends as a metric for maze generators [#ref buck2015mazes].

This effect occurs because tiles that would otherwise form part of a single long passage are instead used as dead-ends, which are largely irrelevant for pathfinding — only dead-ends located directly at the start or end of a path affect its length. Consequently, path length per unit maze area decreases slightly as maze area increases, as illustrated in [#img pathlengths_per_area.png].

*(Figure 3: Path lengths per unit area (UA) decrease as maze area increases.)*

Additionally, the minimum path length between safe zones and survivors decreases as the number of safe zones increases, since more short and efficient paths become available. At the same time, maximum path lengths increase slightly, due to a higher likelihood of encountering long, suboptimal paths. Nevertheless, overall rescue efficiency improves with a greater number of safe zones, owing to the increased availability of shorter, optimal paths.

### 3.3 Total Distance Traveled by Robots

The total distance traveled by the robots increases with the height and width of the maze, as the distance between safe zones and survivors generally grows. While the path-shortening effect of dead-ends is measurable, it is negligible compared to the effect of increasing maze area.

The total distance traveled also increases with the number of survivors, though the slope flattens slightly for larger survivor counts, due to decreasing minimum path lengths between safe zones and survivors.

Increasing the number of safe zones leads to a clear and measurable reduction in the distance traveled by robot agents, as illustrated in [#img tiles_moved_per_survivor_on_safezones.png]. Additional safe zones therefore meaningfully optimize search paths, offering substantial operational advantages for real-world deployment.

*(Figure 4: Total distance traveled per survivor, grouped by safe zone count. A clear inverse relationship between the number of safe zones and the distance moved by robot agents is visible.)*

Conversely, an increasing number of robots results in a slight increase in distance traveled per survivor. This phenomenon is attributed to interference between agents, which disrupts each other's optimal paths to survivors. Although agents share the common goal of rescuing all survivors as quickly as possible, they act competitively rather than cooperatively, since each agent's search for the next survivor disregards the position and path choices of other agents [#ref russell2002artificial]. This constitutes a resource conflict, as previously discussed by Yan et al. in the context of shared-object manipulation [#ref yan2013survey]. Introducing a shared knowledge base as a communication interface would likely mitigate this negative effect.

For real-world implementation, the following recommendations can be derived from these findings: a sufficient number of rescue zones should be established; adequate communication and coordination must be ensured among rescue robots to avoid resource conflicts; and independent agent groups within an MMRS should operate without interfering with one another.

### 3.4 Influence of Robot Count on Simulation Steps

The number of simulation steps naturally decreases as the number of robots increases, since multiple robots can rescue survivors simultaneously. In this context, doubling the number of robots roughly halves the number of required simulation steps, provided a sufficiently large number of survivors is present.

When the number of robots is very high — or exceeds the number of survivors — the number of simulation steps approaches a fixed value of four, corresponding to the minimum sequence of required actions (locate, pick up, transport, drop off) for a single robot rescuing a single survivor. In real-world rescue scenarios, minimizing the number of process steps is therefore critical for accelerating survivor rescue; the number of available rescue units is a key factor in achieving this.

## 4 Summary and Outlook

This report introduced an agent-based simulation of quadruped rescue robots navigating maze-like disaster environments, implemented in Python using the Mesa framework. Robot navigation relies on the A* algorithm with a Manhattan-distance heuristic, and mazes are generated using an iterative randomized DFS algorithm that produces realistic, dead-end-rich structures.

An exhaustive parameter study comprising 58,800 simulation runs revealed several fundamental relationships: the number of simulation steps depends primarily on the number of survivors and robots rather than on maze dimensions or safe-zone count; relative path length decreases as maze area increases due to the diminishing relevance of dead-ends; total distance traveled grows with maze size and survivor count but decreases substantially with a higher number of safe zones; and increasing the number of robots reduces the required simulation steps but slightly increases per-survivor distance traveled due to uncoordinated interference between agents.

These findings underscore the importance of adequate safe-zone provisioning and effective agent coordination for efficient disaster rescue operations. Future work will build on this simulation model to examine the optimal placement of safe zones in greater detail and to discuss the requirements and feasibility of a real-world implementation using quadruped robots, such as Boston Dynamics' Spot.

## References

*(Numbered in order of appearance in the text, IEEE citation style; verify against `references.bib` before submission.)*

[1] A. Dorri, S. S. Kanhere, and R. Jurdak, "Multi-Agent Systems: A Survey," *IEEE Access*, vol. 6, pp. 28573–28593, 2018.

[2] D. S. Drew, "Multi-Agent Systems for Search and Rescue Applications," *Current Robotics Reports*, vol. 2, no. 2, pp. 189–200, 2021.

[3] RoboCup Rescue Robot League Committee, "RoboCupRescue Robot League," 2025. [Online]. Available: <https://rrl.robocup.org/> [Accessed: 26 Sep 2025].

[4] A. Visser, N. Ito, and A. Kleiner, "RoboCup Rescue Simulation Innovation Strategy," in *RoboCup 2014: Robot World Cup XVIII*, Cham: Springer International Publishing, 2015, pp. 661–672.

[5] Z. Yan, N. Jouandeau, and A. A. Cherif, "A Survey and Analysis of Multi-Robot Coordination," *International Journal of Advanced Robotic Systems*, vol. 10, no. 12, p. 399, 2013.

[6] L. E. Parker, D. Rus, and G. S. Sukhatme, "Multiple Mobile Robot Systems," in *Springer Handbook of Robotics*, Cham: Springer International Publishing, 2016, pp. 1335–1384.

[7] P. E. Hart, N. J. Nilsson, and B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," *IEEE Transactions on Systems Science and Cybernetics*, vol. 4, no. 2, pp. 100–107, 1968.

[8] E. W. Dijkstra, "A Note on Two Problems in Connexion with Graphs," *Numerische Mathematik*, vol. 1, no. 1, pp. 269–271, 1959.

[9] P.-O. Siebers and U. Aickelin, "Introduction to Multi-Agent Simulation," in *Encyclopedia of Decision Making and Decision Support Technologies*, IGI Global, 2008, pp. 554–564.

[10] R. E. Korf, "Depth-First Iterative-Deepening: An Optimal Admissible Tree Search," *Artificial Intelligence*, vol. 27, no. 1, pp. 97–109, 1985.

[11] E. Gelenbe and F.-J. Wu, "Large Scale Simulation for Human Evacuation and Rescue," *Computers and Mathematics with Applications*, vol. 64, no. 12, pp. 3869–3880, 2012.

[12] A. Hagberg, P. J. Swart, and D. A. Schult, "Exploring Network Structure, Dynamics, and Function using NetworkX," Los Alamos National Laboratory (LANL), Los Alamos, NM, 2008.

[13] A. Patel, "Heuristics – From Amit's Thoughts on Pathfinding," 2025. [Online]. Available: <https://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html> [Accessed: 23 Sep 2025].

[14] E. ter Hoeven et al., "Mesa 3: Agent-Based Modeling with Python in 2025," *Journal of Open Source Software*, vol. 10, no. 107, p. 7668, 2025.

[15] M. L. Waskom, "seaborn: Statistical Data Visualization," *Journal of Open Source Software*, vol. 6, no. 60, p. 3021, 2021.

[16] W. Kirch, "Pearson's Correlation Coefficient," in *Encyclopedia of Public Health*, Dordrecht: Springer Netherlands, 2008, pp. 1090–1091.

[17] V. Bellot et al., "How to Generate Perfect Mazes?," *Information Sciences*, vol. 572, pp. 444–459, 2021.

[18] J. Buck, *Mazes for Programmers: Code Your Own Twisty Little Passages*. Dallas, TX: The Pragmatic Bookshelf, 2015.

[19] S. J. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 2nd ed. Upper Saddle River, NJ: Prentice Hall, 2002.
