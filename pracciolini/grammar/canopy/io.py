from pracciolini.core.decorators import load, save
from pracciolini.grammar.canopy.model.subgraph import SubGraph


@load("canopy_subgraph", ".cnpy")
def load_canopy_subgraph(file_path: str) -> SubGraph:
    subgraph: SubGraph = SubGraph.load(file_path)
    return subgraph


@save("canopy_subgraph", ".cnpy")
def save_canopy_subgraph(subgraph: SubGraph, file_path: str) -> str:
    subgraph.save(file_path)
    return file_path

