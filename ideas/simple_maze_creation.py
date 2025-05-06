from pyamaze import maze, agent

# m = maze(10, 10)
# m.CreateMaze(theme=COLOR.light)

m = maze(20, 20)
m.CreateMaze(loopPercent=20, pattern="h")

a = agent(m, footprints=True)
# a.position=(5,4)
# a.position=(5,3)
# a.position=(5,2)
m.tracePath({a: m.path}, delay=500)
m.run()

print(m.maze_map)
print(m.cols, m.rows)
print(m.path)
