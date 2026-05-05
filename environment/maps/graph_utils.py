# https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.successors.html
# https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.out_edges.html

import json
from pathlib import Path
import networkx as nx

def generate_graph(map_path: str | Path = None) -> nx.Graph:
    if map_path is None:
        map_path = Path(__file__).parent / "classic.json"

    with open(map_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    territories = data["territories"]
    continents  = data["continents"]

    terr_to_cont = {}
    for cont in continents:
        for t in cont["territories"]:
            terr_to_cont[t] = cont["name"]

    G = nx.Graph()

    G.graph["node_to_idx"] = {}
    G.graph["idx_to_node"] = {}

    for idx, (territory_name, attrs) in enumerate(territories.items()):
        G.add_node(
            territory_name,
            id=idx,
            continent=terr_to_cont[territory_name]
        )
        G.graph["node_to_idx"][territory_name] = idx
        G.graph["idx_to_node"][idx] = territory_name

    for territory_name, attrs in territories.items():
        for nbr_name in attrs["adjacency"]:
            G.add_edge(territory_name, nbr_name)

    G.graph["continents"] = {
        cont["name"]: {"bonus": cont["bonus"], "territories": cont["territories"]}
        for cont in continents
    }

    G = G.to_directed()
    G.graph["edge_to_idx"] = {}
    G.graph["idx_to_edge"] = {}
    for i, (u, v) in enumerate(list(G.edges())):
        G.graph["edge_to_idx"][(u, v)] = i
        G.graph["idx_to_edge"][i] = (u, v)

    for i, (u, v) in enumerate(list(G.edges())):
        G[u][v]["id"] = i

    def edge_to_nodes_id(self, idx):
        edge = self.graph["idx_to_edge"]
        src, dst = edge
        return self.graph["node_to_idx"][src], self.graph["node_to_idx"][dst]

    G.edge_to_nodes_id = edge_to_nodes_id

    return G