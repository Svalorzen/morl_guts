"""
In this experiment, we test having reference points [0,..,0] and [1,..,1]
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')
from gp_utilities import utils_experiment, utils_parameters

start_seed = 13
num_queries = 25
for num_iter in [1, 10, 25, 50, 100]:
    plt.figure(figsize=(10, 4))

    plt_idx = 1
    for num_obj in [5]:
        for noise_level in [0.01]:

            plt.title('{} obj., {} noise, avg over {}'.format(num_obj, noise_level, num_iter))

            for virtcomps in ['none', 'VC grid', 'VC pcs', 'VC grid begin', 'VC pcs begin']:

                # 5 is the number of diff query types;
                # 50 is the number of queries we ask
                utl_vals = np.zeros((5 * num_iter, 50))
                iter_idx = 0

                for query_type in ['pairwise', 'clustering', 'ranking', 'top_rank']:

                    params = utils_parameters.get_parameter_dict(query_type=query_type, num_objectives=num_obj, utility_noise=noise_level)
                    if virtcomps == 'VC grid':
                        params['VC grid'] = True
                    elif virtcomps == 'VC pcs begin':
                        params['VC grid begin'] = True

                    elif virtcomps == 'VC pcs':
                        params['VC pcs'] = True
                    elif virtcomps == 'VC pcs begin':
                        params['VC pcs begin'] = True

                    plt.title(query_type)

                    params['seed'] = start_seed

                    for _ in range(num_iter):

                        experiment = utils_experiment.Experiment(params)

                        result = experiment.run(recalculate=False)

                        utl_vals[iter_idx] = result[0]

                        params['seed'] += 1
                        iter_idx += 1

                if virtcomps == 'none':
                    plt.plot(np.mean(utl_vals, axis=0), '-', label='none')
                elif virtcomps == 'VC grid':
                    plt.plot(np.mean(utl_vals, axis=0), '--', label='VC grid')
                elif virtcomps == 'VC pcs':
                    plt.plot(np.mean(utl_vals, axis=0), '.', label='VC pcs')
                elif virtcomps == 'VC grid begin':
                    plt.plot(np.mean(utl_vals, axis=0), '--', label='VC grid begin')
                elif virtcomps == 'VC pcs begin':
                    plt.plot(np.mean(utl_vals, axis=0), '.', label='VC pcs begin')

            plt.gca().set_ylim(top=1)
            plt_idx += 1

    plt.legend()
    plt.tight_layout()
    plt.savefig('result_plots/mono_virtcomps_{}'.format(num_iter))
    plt.close()
