"""This script evaluates various strategies for handling a misprediction
(varying the number of downstream decode tasks that we reset upon realizing that
a prior dependency was decoded incorrectly). It saves the results to a file for
later analysis.

This script takes under five minutes to run on an M1 Macbook Pro."""

import os, sys
sys.path.append('.')
import json
import networkx as nx
from swiper.predictor import strategy_sim

if __name__ == '__main__':
    d=13
    num_nodes = 100
    num_shots = 10_000
    decode_times = [2 * 13, 5 * 13, 10 * 13]
    test_graph = nx.DiGraph()
    test_graph.add_nodes_from([(node, {'t': d * node}) for node in range(num_nodes)])
    test_graph.add_edges_from([(i, i+1) for i in range(num_nodes - 1)])
    adj_pairs = list(zip(list(test_graph.edges())[:-1], list(test_graph.edges())[1:]))
    opt_runtimes = {decode_time: [] for decode_time in decode_times}
    opt_classicals = {decode_time: [] for decode_time in decode_times}
    opt_procs = {decode_time: [] for decode_time in decode_times}
    pes_runtimes = {decode_time: [] for decode_time in decode_times}
    pes_classicals = {decode_time: [] for decode_time in decode_times}
    pes_procs = {decode_time: [] for decode_time in decode_times}
    adj_runtimes = {decode_time: [] for decode_time in decode_times}
    adj_classicals = {decode_time: [] for decode_time in decode_times}
    adj_procs = {decode_time: [] for decode_time in decode_times}

    for decode_time in decode_times:
        for _ in range(num_shots):
            opt_runtime, opt_classical, opt_valid, opt_proc = strategy_sim(test_graph, adj_pairs, decode_time, 1, 0.9, 0.95, 'optimistic')
            assert opt_valid == num_nodes * decode_time
            pes_runtime, pes_classical, pes_valid, pes_proc = strategy_sim(test_graph, adj_pairs, decode_time, 1, 0.9, 0.95, 'pessimistic')
            assert pes_valid == num_nodes * decode_time
            adj_runtime, adj_classical, adj_valid, adj_proc = strategy_sim(test_graph, adj_pairs, decode_time, 1, 0.9, 0.95, 'adjacent')
            assert adj_valid == num_nodes * decode_time
            opt_runtimes[decode_time].append(opt_runtime)
            opt_classicals[decode_time].append(opt_classical)
            opt_procs[decode_time].append(opt_proc)
            pes_runtimes[decode_time].append(pes_runtime)
            pes_classicals[decode_time].append(pes_classical)
            pes_procs[decode_time].append(pes_proc)
            adj_runtimes[decode_time].append(adj_runtime)
            adj_classicals[decode_time].append(adj_classical)
            adj_procs[decode_time].append(adj_proc)
    
    with open('artifact/data/mispredict_data.json', 'w') as f:
        json.dump({
            'd': d,
            'num_nodes': num_nodes,
            'num_shots': num_shots,
            'decode_times': decode_times,
            'opt_runtimes': opt_runtimes,
            'opt_classicals': opt_classicals,
            'opt_procs': opt_procs,
            'pes_runtimes': pes_runtimes,
            'pes_classicals': pes_classicals,
            'pes_procs': pes_procs,
            'adj_runtimes': adj_runtimes,
            'adj_classicals': adj_classicals,
            'adj_procs': adj_procs,
        }, f)