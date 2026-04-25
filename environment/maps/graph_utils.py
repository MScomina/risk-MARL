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

    for tid, attrs in territories.items():
        G.add_node(
            tid,
            continent=terr_to_cont[tid],
        )

    for tid, attrs in territories.items():
        for nbr_name in attrs["adjacency"]:
            G.add_edge(tid, nbr_name)

    G.graph["continents"] = {
        cont["name"]: {"bonus": cont["bonus"], "territories": cont["territories"]}
        for cont in continents
    }

    return G