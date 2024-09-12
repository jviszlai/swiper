import networkx as nx
import numpy as np

class surface_patch():
    '''
    Minimal representation of a surface code for a program decoding trace
    '''
    def __init__(self, coords: tuple[int, int]) -> None:
        self.coords = coords

class lattice_surgery_op():
    '''
    A tree-like lattice surgery operation whose decoding dependency graph is guaranteed to be 2-colorable
    '''
    
    def __init__(self, data_patches: list[surface_patch], active_data_patches: list[surface_patch], routing_patches: list[surface_patch]) -> None:
        self.data_patches = data_patches
        self.active_coords = [patch.coords for patch in active_data_patches + routing_patches]
        self.routing_patches = routing_patches
    
    def gen_subgraph(self, t0: int = 0, delays: int = 0) -> nx.Graph:
        G = nx.Graph()
        for patch in self.data_patches:
            G.add_node(patch.coords + (t0,), data=True)
        for patch in self.routing_patches:
            G.add_node(patch.coords + (t0,), data=False)
            for coords in G.nodes:
                if (coords[0], coords[1]) in self.active_coords and np.linalg.norm(np.array(patch.coords + (t0,)) - np.array(coords)) == 1:
                    G.add_edge(patch.coords + (t0,), coords)
        for t in range(delays):
            for patch in self.data_patches:
                G.add_node(patch.coords + (t0 + t + 1,), data=True)
                G.add_edge(patch.coords + (t0 + t,), patch.coords + (t0 + t + 1,))
        return G
