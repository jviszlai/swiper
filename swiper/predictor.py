import stim
import pymatching
import numpy as np
import pickle as pkl
import datetime
import networkx as nx

def speculator(samples, weight_1_chains, weight_2_chains, ignore_step=[], one_step_chains=[]):
    shared_syndromes = samples
    output_syndromes = shared_syndromes.copy()
    believed_match = [[False for _ in range(len(weight_1_chains))] for _ in range(len(samples))]
    matchings = [[] for _ in range(len(samples))]
    def step1(i, chain, match=False):
        for j, shared_sample in enumerate(shared_syndromes):
            if len(chain) == 1:
                if shared_sample[chain[0]]:
                    output_syndromes[j][chain[0]] += 50
                    believed_match[j][i] = True
            else:
                if shared_sample[chain[0]] and shared_sample[chain[1]]:
                    output_syndromes[j][chain[0]] += 1
                    output_syndromes[j][chain[1]] += 1
                    believed_match[j][i] = True
                    if match:
                        matchings[j].append(chain)
        return
    def step2(i, j, chain):
        if len(chain) == 1:
            if output_syndromes[j][chain[0]] > 0:
                output_syndromes[j][chain[0]] = 0
                matchings[j].append(chain)
        else:
            if output_syndromes[j][chain[0]] > 0 and output_syndromes[j][chain[1]] > 0:
                output_syndromes[j][chain[0]] = 0
                output_syndromes[j][chain[1]] = 0
                matchings[j].append(chain)
    def step3(i, chain):
        for j, shared_sample in enumerate(shared_syndromes):
            if shared_sample[chain[0]] and shared_sample[chain[1]]:
                matchings[j].append(chain)
        return
    
    if 1 not in ignore_step:
        if [2,3] == ignore_step:
            for i, weight_1_chain in enumerate(one_step_chains):
                step1(i, weight_1_chain, match=True)
        else:
            for i, weight_1_chain in enumerate(weight_1_chains):
                step1(i, weight_1_chain)

        shared_syndromes = output_syndromes
        output_syndromes = shared_syndromes.copy()
    if 2 not in ignore_step:
        for j, shared_sample in enumerate(shared_syndromes):
            sorted_1_indices = sorted(range(len(weight_1_chains)), key=lambda i: shared_sample[weight_1_chains[i][0]] + (shared_sample[weight_1_chains[i][1]] if len(weight_1_chains[i]) > 1 else 100))
            for i in sorted_1_indices:
                step2(i, j, weight_1_chains[i])
        shared_syndromes = output_syndromes
        output_syndromes = shared_syndromes.copy()
    if 3 not in ignore_step:
        for i, weight_2_chain in enumerate(weight_2_chains):
            step3(i, weight_2_chain)

    return matchings
    

def extract_data_deps(syndrome, matching, coords_dict, boundary_idx, boundary_ub, nx_G, boundary_node=-1):
    dep_matchings = {}
    output_syndrome = syndrome.copy()
    boundary_dets = [det for det, coords in coords_dict.items() if coords[boundary_idx] == boundary_ub]
    det_lookup = {tuple(coords): det for det, coords in coords_dict.items()}
    
    for chain in matching:
        boundary_match = None
        if len(chain) == 1:
            boundary_match = chain[0]
        elif chain[0] == -1:
            boundary_match = chain[1]
        elif chain[1] == -1:
            boundary_match = chain[0] 
        if boundary_match:
            continue
        det1, det2 = chain
        if det1 == -1 or det2 == -1:
            continue
        t_coord1 = coords_dict[det1][boundary_idx]
        t_coord2 = coords_dict[det2][boundary_idx]
        min_t, max_t = min(t_coord1, t_coord2), max(t_coord1, t_coord2)
        if min_t < boundary_ub and max_t >= boundary_ub:
            if t_coord1 < t_coord2:
                dep_matchings[det1] = det2
            else:
                dep_matchings[det2] = det1
            if t_coord2 == max_t:
                artificial_det = det_lookup[tuple(coords_dict[det2][:boundary_idx]) + (boundary_ub,) + tuple(coords_dict[det2][boundary_idx+1:])]
                output_syndrome[artificial_det] ^= True
            elif t_coord1 == max_t:
                artificial_det = det_lookup[tuple(coords_dict[det1][:boundary_idx]) + (boundary_ub,) + tuple(coords_dict[det1][boundary_idx+1:])]
                output_syndrome[artificial_det] ^= True
                
    return [output_syndrome[det] for det in boundary_dets], dep_matchings

def verify_speculation(syndrome, spec_matches, real_matches, coords_dict, boundary_idx, boundary_ub, nx_G):
    spec_syndrome, spec_deps  = extract_data_deps(syndrome, spec_matches, coords_dict, boundary_idx, boundary_ub, nx_G, len(coords_dict))
    real_syndrome, real_deps  = extract_data_deps(syndrome, real_matches, coords_dict, boundary_idx, boundary_ub, nx_G)
    return spec_syndrome == real_syndrome, spec_syndrome, real_syndrome


def simulate_temporal_speculation(num_shots, d, p, ignore_steps=[]):
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", 
                              distance=d, rounds=2*d,
                              after_clifford_depolarization=p, 
                              before_measure_flip_probability=p,
                              )
    coords_dict = circ.detector_error_model().get_detector_coordinates()
    matching = pymatching.Matching.from_detector_error_model(circ.detector_error_model())
    boundary_node = len(coords_dict)

    sampler = circ.compile_detector_sampler()
    temporal_boundary_mask = [False for _ in range(len(coords_dict))]
    for det, coords in coords_dict.items():
        if coords[2] >= d - 2 and coords[2] <= d + 2:
            temporal_boundary_mask[det] = True
    temporal_boundary_mask = np.array(temporal_boundary_mask)

    nx_G = matching.to_networkx()
    edges_to_remove = []
    nodes_to_remove = np.where(temporal_boundary_mask == False)[0]
    for edge in nx_G.edges():
        if edge[0] == boundary_node:
            if not temporal_boundary_mask[edge[1]]:
                edges_to_remove.append(edge)
            continue
        elif edge[1] == boundary_node:
            if not temporal_boundary_mask[edge[0]]:
                edges_to_remove.append(edge)
            continue
        elif not temporal_boundary_mask[edge[0]] or not temporal_boundary_mask[edge[1]]:
            edges_to_remove.append(edge)
    nx_G.remove_edges_from(edges_to_remove)
    nx_G.remove_nodes_from(nodes_to_remove)
            
    det_dists = dict(nx.all_pairs_shortest_path_length(nx_G))
    def get_dep_weight(det1, det2):
        return det_dists[det1][det2]

    weight_1_chains = []
    one_step_chains = []
    for det1, det2 in nx_G.edges():
        if det1 == boundary_node:
            if temporal_boundary_mask[det2]:
                weight_1_chains.append((det2,))
            continue
        if det2 == boundary_node:
            if temporal_boundary_mask[det1]:
                weight_1_chains.append((det1,))
            continue
        # Edge therefore weight-1 chain
        if temporal_boundary_mask[det1] and temporal_boundary_mask[det2]:
            weight_1_chains.append((det1, det2))
            t_coord1 = coords_dict[det1][2]
            t_coord2 = coords_dict[det2][2]
            min_t, max_t = min(t_coord1, t_coord2), max(t_coord1, t_coord2)
            if min_t == d - 1 and max_t == d:
                one_step_chains.append((det1, det2))

    data_dep_srcs = []
    for det_id in np.where(temporal_boundary_mask)[0]:
        if coords_dict[det_id][2] < d:
            data_dep_srcs.append(det_id)
    weight_2_chains = []
    for det1 in data_dep_srcs:
        for det2 in np.where(temporal_boundary_mask)[0]:
            if coords_dict[det2][2] >= d and get_dep_weight(det1, det2) == 2:
                weight_2_chains.append((det1, det2))

    match_times = []
    corrects = []
    matching.decode_to_matched_dets_array(sampler.sample(1))

    samples = sampler.sample(num_shots)

    speculated_matchings = speculator(samples.astype(int), weight_1_chains, weight_2_chains, ignore_steps, one_step_chains)
    failure_idx = []
    for i in range(num_shots):
        match_time = datetime.datetime.now()
        matched_dets = matching.decode_to_matched_dets_array(samples[i])
        match_time = (datetime.datetime.now() - match_time).total_seconds() * 1e6 # us
        correct, spec_syndrome, real_syndrome = verify_speculation(samples[i], speculated_matchings[i], matched_dets, coords_dict, 2, d, nx_G)
        if not correct:
            failure_idx.append((samples[i], spec_syndrome, real_syndrome, matched_dets))
        match_times.append(match_time)
        corrects.append(correct)
    return match_times, corrects, (matching.to_networkx(), coords_dict, failure_idx)

def process_failures(failure_info, d):
    coords_dict = failure_info[1]
    boundary_dets = [det for det, coords in coords_dict.items() if coords[2] == d]
    false_neg = 0
    false_pos = 0
    both = 0
    for i, (sample, spec_syndrome, real_syndrome, real_matches) in enumerate(failure_info[2]):
        original_sample = [sample[det] for det in boundary_dets]
        is_false_neg = False
        is_false_pos = False
        for j, val in enumerate(original_sample):
            if val and not real_syndrome[j] and spec_syndrome[j]:
                is_false_neg = True
            if val and real_syndrome[j] and not spec_syndrome[j]:
                is_false_pos = True
            if not val and real_syndrome[j] and not spec_syndrome[j]:
                is_false_neg = True
            if not val and real_syndrome[j] and not spec_syndrome[j]:
                is_false_pos = True
        if is_false_pos and is_false_neg:
            both += 1
        elif is_false_neg:
            false_neg += 1
        elif is_false_pos:
            false_pos += 1
    return false_neg, false_pos, both


def strategy_sim(nx_G: nx.DiGraph, adj_pairs: tuple[tuple[int,int]], decode_time: int, spec_time: int, accuracy: float, cond_mult: float, strategy: str) -> tuple[int, int]:
    decoding_queue = {}
    start_times = nx.get_node_attributes(nx_G, 't')
    for node in nx_G.nodes:
        if start_times[node] not in decoding_queue:
            decoding_queue[start_times[node]] = []
        decoding_queue[start_times[node]].append(node)
    pred_accuracy = {boundary: accuracy for boundary in nx_G.edges}
    active_rounds = {}
    active_nodes = {}
    complete_nodes = {}
    decode_rounds = 0
    valid_rounds = 0
    max_proc = 0
    round = 0
    while len(decoding_queue) > 0 or len(active_nodes) > 0:
        decode_rounds += len(active_nodes)
        max_proc = max(max_proc, len(active_nodes))
        if round + decode_time not in active_rounds:
            active_rounds[round + decode_time] = []
        if round + decode_time + spec_time not in active_rounds:
            active_rounds[round + decode_time + spec_time] = []
        to_commit = active_rounds[round].copy() if round in active_rounds else []
        for node in to_commit:
            if node not in active_rounds[round]:
                continue
            active_rounds[round].remove(node)
            active_nodes.pop(node)
            complete_nodes[node] = round
            valid_rounds += decode_time
            source_boundries = nx_G.out_edges(node)
            for (_, dependent) in source_boundries:
                boundary = (node, dependent)
                pred = np.random.choice([True, False], p=[pred_accuracy[boundary], 1 - pred_accuracy[boundary]])
                if not pred:
                    if dependent in active_nodes:
                        active_rounds[active_nodes[dependent]].remove(dependent)
                        active_nodes.pop(dependent)
                        active_rounds[round + decode_time].append(dependent)
                        active_nodes[dependent] = round + decode_time
                    elif dependent in complete_nodes:
                        complete_nodes.pop(dependent)
                        valid_rounds -= decode_time
                        active_rounds[round + decode_time].append(dependent)
                        active_nodes[dependent] = round + decode_time
                    if strategy == 'pessimistic':
                        descendants = nx.descendants(nx_G, dependent)
                        for restart_node in descendants:
                            if restart_node in active_nodes:
                                active_rounds[active_nodes[restart_node]].remove(restart_node)
                                active_nodes.pop(restart_node)
                                active_nodes[restart_node] = round + spec_time + decode_time
                                active_rounds[round + decode_time + spec_time].append(restart_node)
                            elif restart_node in complete_nodes:
                                complete_nodes.pop(restart_node)
                                valid_rounds -= decode_time
                                active_nodes[restart_node] = round + spec_time + decode_time
                                active_rounds[round + decode_time + spec_time].append(restart_node)
                    elif strategy == 'optimistic':
                        adj_edges = nx_G.out_edges(dependent)
                        for adj_edge in adj_edges:
                            if (boundary, adj_edge) in adj_pairs:
                                pred_accuracy[adj_edge] = accuracy * cond_mult
                    elif strategy == 'adjacent':
                        adj_edges = nx_G.out_edges(dependent)
                        for adj_edge in adj_edges:
                            if (boundary, adj_edge) in adj_pairs:
                                restart_node = adj_edge[1]
                                if restart_node in active_nodes:
                                    active_rounds[active_nodes[restart_node]].remove(restart_node)
                                    active_nodes.pop(restart_node)
                                    active_nodes[restart_node] = round + spec_time + decode_time
                                    active_rounds[round + decode_time + spec_time].append(restart_node)
                                elif restart_node in complete_nodes:
                                    complete_nodes.pop(restart_node)
                                    valid_rounds -= decode_time
                                    active_nodes[restart_node] = round + spec_time + decode_time
                                    active_rounds[round + decode_time + spec_time].append(restart_node)
                pred_accuracy[boundary] = 1
        to_schedule = decoding_queue[round] if round in decoding_queue else []
        for node in to_schedule:
            decoding_queue[round].remove(node)
            if len(decoding_queue[round]) == 0:
                decoding_queue.pop(round)
            active_rounds[round + decode_time].append(node)
            active_nodes[node] = round + decode_time
            
        round += 1
    return round, decode_rounds, valid_rounds, max_proc