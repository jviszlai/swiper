from swiper.dep_graph import di_dependency_graph
import networkx as nx
import numpy as np

class DecodingSim():

    def __init__(self, t_w: int, pred_acc: int) -> None:
        self.t_w = t_w
        self.pred_acc = pred_acc
    
    def run(self, di_dep_graph: di_dependency_graph, pred: bool=True, blocking_cycles: list[int]=[]) -> dict:
        
        def gen_sched(graph):
            cycle_sched = {}
            for node in graph.di_dep_graph.nodes:
                if node[2] not in cycle_sched:
                    cycle_sched[node[2]] = []
                cycle_sched[node[2]].append(node)
            return cycle_sched

        cycle_sched = gen_sched(di_dep_graph)
            
        waiting_nodes = []
        waiting_trace = []
        decoded_sched = {i: [] for i in range(self.t_w)}
        decoded_trace = {}
        uncommitted_nodes = list(di_dep_graph.di_dep_graph.nodes)
        committed_nodes = []

        predicted_nodes = []
        next_blocking_cycle = blocking_cycles.pop(0)
        cycle = 0
        while len(uncommitted_nodes) > 0:
            if cycle in cycle_sched:
                waiting_nodes.extend(cycle_sched[cycle])
                predicted_nodes.extend(cycle_sched[cycle])
            decoded_sched[cycle + self.t_w] = []

            for node in decoded_sched[cycle].copy():
                if node in waiting_nodes:
                    continue
                uncommitted_nodes.remove(node)
                committed_nodes.append(node)
                if pred:
                    pred_success = np.random.randint(0, 100_000) < self.pred_acc
                    if not pred_success:
                        poisoned_nodes = [node for node in nx.descendants(di_dep_graph.di_dep_graph, node)]
                        for poisoned_node in poisoned_nodes:
                            if poisoned_node in predicted_nodes:
                                if poisoned_node not in waiting_nodes:
                                    waiting_nodes.append(poisoned_node)
                                if poisoned_node not in uncommitted_nodes:
                                    uncommitted_nodes.append(poisoned_node)
                                    committed_nodes.remove(poisoned_node)
                                for i in decoded_sched:
                                    if poisoned_node in decoded_sched[i]:
                                        decoded_sched[i].remove(poisoned_node)
            for node in waiting_nodes.copy():
                deps = True
                for edge in di_dep_graph.di_dep_graph.in_edges(node):
                    if pred and edge[0] not in predicted_nodes:
                        deps = False
                        break
                    elif not pred and edge[0] in uncommitted_nodes:
                        deps = False
                        break
                if deps:
                    decoded_sched[cycle + self.t_w].append(node)
                    waiting_nodes.remove(node)

            if cycle >= next_blocking_cycle:
                add_stall = False
                for node in uncommitted_nodes:
                    if node[2] <= next_blocking_cycle:
                        add_stall = True
                if add_stall:
                    blocking_cycles = [c + 1 for c in blocking_cycles]
                    di_dep_graph.add_stall(cycle)
                    uncommitted_nodes = list(di_dep_graph.di_dep_graph.nodes)
                    for node in committed_nodes:
                        uncommitted_nodes.remove(node)
                    cycle_sched = gen_sched(di_dep_graph)
                else:
                    if len(blocking_cycles) > 0:
                        next_blocking_cycle = blocking_cycles.pop(0)
                    
            waiting_trace.append(len(waiting_nodes))
            decoded_trace[cycle] = len(committed_nodes)
            cycle += 1

        
        return waiting_trace, decoded_trace
                
