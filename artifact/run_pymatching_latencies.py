"""This script collects PyMatching latencies for randomly generated surface code
decoding problems of varying distance and number of rounds, then saves them to a
file for use in the benchmark simulations.

On an M1 Macbook Pro, this script takes about 1 hour to run."""

import os, sys
sys.path.append('.')
import stim
import pymatching
import numpy as np
import datetime
import json
import pickle as pkl

if __name__ == '__main__':
    num_shots = 10_000
    d_range = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    decoding_dists = {d: {} for d in d_range}
    p = 1e-3

    print(f"Running pymatching decoder latencies for d in {d_range} and p = {p}")
    for d in d_range:
        print(f"Running d = {d}, r = ", end='', flush=True)
        for r in range(2, 8):
            print(f"{r} ", end='', flush=True)
            circ = stim.Circuit.generated("surface_code:rotated_memory_z", 
                                    distance=d, rounds=r*d,
                                    after_clifford_depolarization=p, 
                                    before_measure_flip_probability=p,
                                    )
            matching = pymatching.Matching.from_detector_error_model(circ.detector_error_model())
            sampler = circ.compile_detector_sampler()
            shots, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
            # Decode one shot first to ensure internal C++ representation of the matching graph is fully cached
            matching.decode_batch(shots[0:1, :])
            decoding_dists[d][r] = np.zeros(num_shots)
            # Now time decoding the batch
            for i in range(num_shots):
                shot = shots[i:i+1, :]
                t0 = datetime.datetime.now()
                matching.decode(shot)
                t1 = (datetime.datetime.now() - t0).total_seconds() * 1e6 # us
                decoding_dists[d][r][i] = t1
        print()
    
    decoding_dists_listified = {d: {r: decoding_dists[d][r].tolist() for r in decoding_dists[d]} for d in decoding_dists}
    pkl.dump(decoding_dists_listified, open('artifact/data/decoder_dists.pkl', 'wb'))
    with open('artifact/data/decoder_dists.json', 'w') as f:
        json.dump(decoding_dists_listified, f)