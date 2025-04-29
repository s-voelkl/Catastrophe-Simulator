import matplotlib.pyplot as plt
import networkx as nx

G = nx.erdos_renyi_graph(20, 0.1)

for i in nx.shortest_path_length(G):
    print(i)
    print(max(list(i[1].values())))

nx.draw(G, with_labels=True)
plt.show()
