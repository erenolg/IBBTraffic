import networkx as nx

def get_graph_topology(month="202301"):

    edge_index_path = f"./data/{month}/edge_index.txt"

    with open(edge_index_path, "r") as f:
        lines = [i.strip() for i in f.readlines()]
        edge_list = []
        for i in lines:
            idxs = [int(j) for j in i.split()]
            edge_list.append(idxs)

    G = nx.Graph(edge_list, directed=False)
    return G
