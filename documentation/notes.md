# Notizen

## Auszug aus dem Kurs

"""
Fällig: Dienstag, 13. Mai 2025, 17:00
Die Dateien  (Quellcode + Bericht + Präsentation) bitte als Archiv-Datei hochladen.

Name der Datei: KI2_Projekt_Name1_Name2.zip

Bericht:

- Einleitung
- Architektur (Klassendiagramm), Beschreibung der Logik
- Ergebnisse und Visualisierung der Simulation
- Quellen

"""

## Aufbau

### Einleitung als Abstract - grundlegender Projektaufbau und erwonnene Erkenntnisse

Vorgehensweise: Angefangen mit der Struktur für Pyamaze (Dict).
Später gemerkt, dass das nicht gut ist -> Visualisierung per Konsole. Zufällig auch Visualisierbarkeit
mit NetworkX per Positioning-Parameter gefunden.
Ansatz als OOP mit Tiles, in einem Environment gespeichert.
Generierung mit random DFS (siehe später), Pfadsuche per Graphen mit Tiles als Nodes mit A-Stern.

Zusammenfassung der Erkenntnisse (KURZ!!!) bei Tests mit jeweils n=500 (?) Durchführungen:

- z.B. Scatter Plot von [Anzahl Survivors] vs. [insgesamt abgelaufene Tiles]
- z.B. Auswirkung Anzahl RobotAgents
- z.B. Auswirkung mit Lifetime der Survivors

### Architektur: Klassendiagramm/ER/OOP, Kernkonzepte (random DFS, A*, Graph, Tiles) erklären

#### Klassendiagramm

Position [mit x,y]
Tile(Position) [mit walls]
Survivor(Tile) [mit is_rescued]
SaveZone(Tile) []
RobotAgent(mesa.Agent) [für 1 Model] [Properties...]
EnvironmentModel(mesa.Model) [1 hat n Survivors, n SaveZones, n RobotAgents] [Properties...]

#### Kernkonzepte

Random DFS
A-Stern
Graph mit Nodes mit dem Typ Tile [x,y, walls]
Survivor Placement
RobotAgent Placement (nur auf SaveZones), Aufgabe (steps)
Mesa EnvironmentModel mit steps, Datensammlung, ...
Jupyter Notebook

## Analyse

### Metriken

Pfadlänge, Dichte, Symmetrie, Ausgänge (inkl. Ansatz)

### Visualisierungen

Probleme mit Pyamaze
Visualisierung mit Console und NetworkX Graph

### Ergebnisse

Plots, Erkenntnisse...
Anzahl Durchführungen (z.B. n=500 ?)

.
