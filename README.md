# Catastrophe-Simulator

Catastrophe Simulator in Python using agents, mazes, graphs and more topics of symbolical AI.

## 1. Labyrinth erstellen -> random DFS (anpasst) o.ä

Random dfs mit List[Tile] -> Dict

## 2. Metriken zum Labyrinth

Environment.get_pathlengths() -> List[int]
Environment.get_min_pathlength(pathlengths: List[int]) -> int
Environment.get_avg_pathlength(pathlengths: List[int]) -> int

- kleinste/mittlere Pfadlänge im Maze. Pfadlänge: Durchschnittliche Entfernung zwischen zwei Knoten (Grad der Trennung).
  vgl.
- Dichte: Verhältnis von Wänden zu offenen Wegen von der gesamten Fläche des Labyrinths.
  "{(1, 1): {'E': 1, 'W': 0, 'N': 0, 'S': 1}...}" -> 1en und 0en speichern
  Environment.\_get_wall_densities() -> List[int]
  Environment.get_mean_wall_density() -> int
- Die Anzahl der Ausgänge und die Symmetries des Labyrinths
  Environment.get_escape_tile_count() -> int. pro äußerem tile nachsehen, ob border wall existiert
  Environment.get_symmetries(). Idee: Spiegeln, Halbes Maze erstellen und das mirroren

## 3. randomisiert einige Überlebenden, die gerettet werden müssten

Environment.place_random_survivors(maze: Dict[tuple, dict], n_survivors: int) -> int (successfully placed survivors)

- forbidden tiles: äußerste rows/cols

## 4. Analyse

Environment.

- wo es am besten wäre für einen Rettungsroboter die Sequenz der Rettungen zu starten, sodass alle Überlebenden am schnellsten gerettet werden könnten. Der Roboter kann nur 1 Person tragen.
- wo der Roboter die Überlebenden am besten raustragen soll
  --> kompletter Pfad

## 5. Ergebnisse visuell darstellen

Environment.display() ??

Environment.show_results() ??

## 6. Überlegen und implementieren Sie eine eigene Erweiterung des Systems
