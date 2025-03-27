import stim
import pymatching
import numpy as np
import datetime
import pickle as pkl

if __name__ == '__main__':
    num_shots = 10_000
    d_range = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    decoding_dists = {d: {} for d in d_range}
    p = 1e-3

    for d in d_range:
        for r in range(2, 8):
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
            
    pkl.dump(decoding_dists, open('artifact/data/decoder_dists.pkl', 'wb'))