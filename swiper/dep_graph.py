import networkx as nx
import numpy as np

class di_dependency_graph():
    '''
    Directed dependency graph for a program decoding trace. The presence of an edge indicates two nodes (decoding windows)
    have an overlapping buffer region. The direction of the edge indicates the forwarding of decoding information. 
    In standard windowed decoding this corresponds to the order in which the nodes are decoded.
    '''

    def __init__(self, decoding_graph: nx.Graph, schedule: str) -> None:
        di_dep_graph = decoding_graph.to_directed()
        if schedule == 'sliding':
            for edge in list(di_dep_graph.edges):
                if edge[0][2] > edge[1][2]:
                    di_dep_graph.remove_edge(*edge)
                elif edge[0][2] == edge[1][2] and (edge[1], edge[0]) in di_dep_graph.edges:
                    di_dep_graph.remove_edge(*edge)
        elif schedule == 'parallel':
            uncolored_nodes = list(di_dep_graph.nodes)
            color_queue = [(uncolored_nodes[0], 0)]
            while len(uncolored_nodes) > 0:
                if len(color_queue) == 0:
                    color_queue.append((uncolored_nodes[0], 0))
                curr_node, color = color_queue.pop(0)
                if curr_node not in uncolored_nodes:
                    continue
                neighbors = list(di_dep_graph.neighbors(curr_node))
                color_queue.extend(zip(neighbors, [color ^ 1] * len(neighbors)))
                for edge in list(di_dep_graph.edges(curr_node)):
                    if edge[0] != curr_node and color == 0:
                        di_dep_graph.remove_edge(*edge)
                    elif edge[1] != curr_node and color == 1:
                        di_dep_graph.remove_edge(*edge)
                uncolored_nodes.remove(curr_node)
        self.di_dep_graph = di_dep_graph
        self.scheduled = schedule
    
    def add_stall(self, t) -> None:
        # Add new time slice @ t and invert all future edges if parallel scheduled
        # Assumes edges between t and t + 1 are only on data (i.e. this is end of d round lattice surgery op)
        new_di_dep_graph = nx.DiGraph()
        for node in list(self.di_dep_graph.nodes):
            if node[2] <= t:
                new_di_dep_graph.add_node(node, data=self.di_dep_graph.nodes[node]['data'])
            if self.di_dep_graph.nodes[node]['data'] and node[2] == t:
                new_di_dep_graph.add_node((node[0], node[1], t + 1), data=True)
            if node[2] > t:
                new_di_dep_graph.add_node((node[0], node[1], node[2] + 1), data=self.di_dep_graph.nodes[node]['data'])
        for node1, node2 in list(self.di_dep_graph.edges):
            min_t_idx = np.argmin([node1[2], node2[2]])
            max_t_idx = np.argmax([node1[2], node2[2]])
            if (node1,node2)[min_t_idx][2] == t and (node1,node2)[max_t_idx][2] == t + 1:
                new_edge1 = ((node1,node2)[min_t_idx], (node1[0], node1[1], t + 1))
                max_t_node = (node1,node2)[max_t_idx]
                new_edge2 = ((node1[0], node1[1], t + 1), (max_t_node[0], max_t_node[1], max_t_node[2] + 1))
                if self.scheduled == 'parallel':
                    new_di_dep_graph.add_edge(new_edge1[0 ^ min_t_idx], new_edge1[1 ^ min_t_idx])
                    new_di_dep_graph.add_edge(new_edge2[0 ^ max_t_idx], new_edge2[1 ^ max_t_idx])
                else:
                    new_di_dep_graph.add_edge(*new_edge1)
                    new_di_dep_graph.add_edge(*new_edge2)
            elif node1[2] <= t and node2[2] <= t:
                new_di_dep_graph.add_edge(node1, node2)
            elif node1[2] >= t:
                new_node1 = (node1[0], node1[1], node1[2] + 1)
                new_node2 = (node2[0], node2[1], node2[2] + 1) 
                if self.scheduled == 'parallel':
                    new_di_dep_graph.add_edge(new_node2, new_node1)
                else:
                    new_di_dep_graph.add_edge(new_node1, new_node2)
        self.di_dep_graph = new_di_dep_graph
        


                