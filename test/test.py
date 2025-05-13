from src.maze import *

test_tile_grid = [
    [
        Tile(x=0, y=0, north=1, east=0, south=0, west=1),
        Tile(x=0, y=1, north=1, east=1, south=1, west=0),
        Tile(x=0, y=2, north=1, east=0, south=1, west=1),
        Tile(x=0, y=3, north=1, east=1, south=0, west=0),
    ],
    [
        Tile(x=1, y=0, north=0, east=0, south=1, west=1),
        Tile(x=1, y=1, north=1, east=1, south=0, west=0),
        Tile(x=1, y=2, north=1, east=0, south=0, west=1),
        Tile(x=1, y=3, north=0, east=1, south=1, west=0),
    ],
    [
        Tile(x=2, y=0, north=1, east=0, south=0, west=1),
        Tile(x=2, y=1, north=0, east=1, south=1, west=0),
        Tile(x=2, y=2, north=0, east=0, south=1, west=1),
        Tile(x=2, y=3, north=1, east=1, south=0, west=0),
    ],
    [
        Tile(x=3, y=0, north=0, east=0, south=1, west=1),
        Tile(x=3, y=1, north=1, east=0, south=1, west=0),
        Tile(x=3, y=2, north=1, east=0, south=1, west=0),
        Tile(x=3, y=3, north=0, east=1, south=1, west=0),
    ],
]


def test_maze():

    m2 = Maze(height=4, width=4)
    print(m2.render_maze())
    m2.tile_grid = test_tile_grid
    print(m2.render_maze())

    m3 = Maze(10,5)
    m3.generate_maze_random_dfs()
    print(m3.render_maze())
    m3.generate_save_zones(3)
    m3.generate_survivors(5)
    print(m3.render_maze("symbol"))
    m3.generate_survivors(5)
    print(m3.render_maze("symbol"))

    m3.visualize_graph()



def main():
    test_maze()


if __name__ == "__main__":
    main()
