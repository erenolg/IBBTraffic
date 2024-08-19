import argparse
from utils import get_graph_topology
from node2vec import Node2Vec
from karateclub import GLEE, NetMF, GraphWave
import pickle

from numpy import triu

def main(args):
    G = get_graph_topology()
    methods = ["node2vec", "glee", "netmf", "graphwave"]
    if args.method not in methods:
        print(f"{args.method} not found! 'node2vec','glee','netmf','graphwave' available.")
        return
    output_path = f"node_embeddings/{args.method}/{args.length}.pkl"
    if args.method.lower() == "node2vec":
        node2vec = Node2Vec(G, dimensions=4, walk_length=30, num_walks=100, workers=4)
        model = node2vec.fit(window=10, min_count=1)
        vector_dict = {node: model.wv[node] for node in G.nodes()}
    elif args.method.lower() == "glee":
        glee = GLEE(3)
        glee.fit(G)
        vectors = glee.get_embedding()
        vector_dict = {node: vectors[node] for node in G.nodes()}
    elif args.method.lower() == "netmf":
        netmf = NetMF(4)
        netmf.fit(G)
        vectors = netmf.get_embedding()
        vector_dict = {node: vectors[node] for node in G.nodes()}
    else:
        pass
    with open(output_path, "wb") as f:
        embeddings = dict(sorted(vector_dict.items(), key=lambda x: x[0]))
        pickle.dump(embeddings, f)
    print(f"{args.method} vectors are created under './node_embeddings/{args.method}/{output_path}'")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")

    parser.add_argument("-t", "--timestamp", type=str, help="Timestamp of snapshot (one timestamp for each hour)")
    parser.add_argument("-m", "--method", type=str, help="Node embedding method", default="netmf")
    parser.add_argument("-l", "--length", type=int, help="Length of node embedding vector", default=4)

    args = parser.parse_args()
    main(args)