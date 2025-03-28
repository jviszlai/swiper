"""This script evaluates the accuracy of the 1-, 2-, and 3-step predictors on
randomly generated syndrome data and saves the results to files for processing.
"""

import pickle as pkl
from swiper.predictor import simulate_temporal_speculation, process_failures

if __name__ == '__main__':
    d_range = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    p_range = [1e-3]

    for i, config in enumerate([[2,3], [3], []]):
        results = {}
        for p in p_range:
            for d in d_range:
                results[(p,d)] = simulate_temporal_speculation(5_000, d, p, ignore_steps=config)
        pkl.dump(results, open(f'artifact/data/0{i+1}-step-predictor-results.pkl', 'wb'))

        print(f'{i+1}-Step Predictor Done')

    one_step_data = pkl.load(open('artifact/data/01-step-predictor-results.pkl', 'rb'))
    two_step_data = pkl.load(open('artifact/data/02-step-predictor-results.pkl', 'rb'))    
    three_step_data = pkl.load(open('artifact/data/03-step-predictor-results.pkl', 'rb'))

    for i, results in enumerate([one_step_data, two_step_data, three_step_data]):
        processed_results = {}
        for p in p_range:
            for d in d_range:
                false_neg, false_pos, both = process_failures(results[(p,d)][2], d=d)
                processed_results[(p,d)] = (results[(p,d)][0], results[(p,d)][1], (false_neg, false_pos, both))
        pkl.dump(processed_results, open(f'artifact/data/processed_0{i+1}-step-predictor-results.pkl', 'wb'))