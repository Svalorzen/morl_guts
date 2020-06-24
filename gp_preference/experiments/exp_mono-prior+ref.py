"""
In this experiment, we test having monotonicity information in terms of
- reference points [0,..,0] and [1,..,1]
- a linear prior
- a mix of the above
"""
import matplotlib
import os
# matplotlib.use('Agg')  # need this for linux
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')
from gp_utilities import utils_experiment, utils_parameters

start_seed = 13
num_queries = 25
for num_iter in [10, 50, 100, 150, 200]:
    plt.figure(figsize=(10.2, 5))

    plt_idx = 1
    for num_obj in [5]:
        for noise_level in [0.01]:

            plt.title('{} obj., {} noise, avg over {}'.format(num_obj, noise_level, num_iter))

            for s in [

                [
                    ['linear-zero', [False, False]],
                    ['linear', [False, False]],
                    ['zero', [False, False]],
                ],

                [
                    ['zero', ['beginning', 'beginning']],
                    ['zero', ['full', 'full']],
                    ['zero', [False, False]],
                ],

                [
                    ['linear-zero', ['full', 'full']],
                    ['linear-zero', [False, False]],
                    ['zero', ['full', 'full']],
                ],

                # [
                #     ['zero', ['beginning', False]],
                #     ['zero', ['full', False]],
                #     ['zero', [False, False]],
                # ],
                #
                # [
                #     ['linear-zero', ['full', False]],
                #     ['linear-zero', [False, False]],
                #     ['zero', ['full', False]],
                # ],

                # [
                #     ['zero', [False, 'beginning']],
                #     ['zero', [False, 'full']],
                #     ['zero', [False, False]],
                # ],
                #
                # [
                #     ['linear-zero', [False, 'full']],
                #     ['linear-zero', [False, False]],
                #     ['zero', [False, 'full']],
                # ],

            ]:
                for [prior_type, reference_points] in s:

                    plt.subplot(1, 3, plt_idx)

                    # 5 is the number of diff query types;
                    # 50 is the number of queries we ask
                    all_query_types = ['pairwise', 'clustering', 'ranking', 'top_rank']
                    utl_vals = np.zeros((len(all_query_types) * num_iter, num_queries))
                    iter_idx = 0

                    for query_type in all_query_types:

                        params = utils_parameters.get_parameter_dict(query_type=query_type, num_objectives=num_obj, utility_noise=noise_level)
                        params['reference min'] = reference_points[0]
                        params['reference max'] = reference_points[1]
                        params['gp prior mean'] = prior_type
                        params['num queries'] = num_queries

                        params['seed'] = start_seed

                        for _ in range(num_iter):

                            experiment = utils_experiment.Experiment(params)
                            result = experiment.run(recalculate=False)
                                    
                            utl_vals[iter_idx] = result[0]
                            params['seed'] += 1
                            iter_idx += 1

                    style = '-'
                    color = 'limegreen'
                    if params['gp prior mean'] == 'zero' and (params['reference min'] == False and params['reference max'] == False):
                        color = 'black'
                        style = '--'
                    elif params['gp prior mean'] == 'linear' and (params['reference min'] == False and params['reference max'] == False):
                        color = 'maroon'
                    elif params['gp prior mean'] == 'linear-zero' and (params['reference min'] == False and params['reference max'] == False):
                        color = 'darkorange'
                    elif params['gp prior mean'] == 'zero' and (params['reference min'] == 'beginning' or params['reference max'] == 'beginning'):
                        color = 'turquoise'
                    elif params['gp prior mean'] == 'zero' and (params['reference min'] == 'full' or params['reference max'] == 'full'):
                        color = 'royalblue'
                    elif params['gp prior mean'] == 'linear-zero' and (params['reference min'] == 'beginning' or params['reference max'] == 'beginning'):
                        color = 'limegreen'
                    else:
                        print("you forgot something....")
                        print(params['gp prior mean'])
                        print(params['reference min'])

                    if params['gp prior mean'] == 'linear-zero':
                        params['gp prior mean'] = 'linear prior (start)'
                    elif params['gp prior mean'] == 'linear':
                        params['gp prior mean'] = 'linear prior (full)'
                    else:
                        params['gp prior mean'] = 'zero prior'

                    if params['reference min'] == 'beginning':
                        params['reference min'] = 'start'
                    if params['reference max'] == 'beginning':
                        params['reference max'] = 'start'

                    if plt_idx == 1:
                        label = '{}'.format(params['gp prior mean'])
                    elif plt_idx == 2:
                        if params['reference min'] or params['reference max']:
                            label = 'ref points ({})'.format(params['reference min']) if params['reference max']==False else 'ref. points ({})'.format(params['reference max'])
                        else:
                            label = 'no ref. points'
                    else:
                        if params['reference min'] != False or params['reference max'] != False:
                            label = '{}; ref points ({})'.format(params['gp prior mean'], params['reference min'])
                        else:
                            label = '{}; no ref points'.format(params['gp prior mean'], params['reference min'])

                    plt.plot(range(1, num_queries+1), np.mean(utl_vals, axis=0), style, color=color, label=label, linewidth=2)

                if plt_idx > 1:
                    plt.yticks([])
                else:
                    plt.ylabel('utility', fontsize=15)

                plt.ylim([0.4, 0.95])
                plt.xlim([1, num_queries])
                plt.xticks([1, 5, 10, 15, 20, 25])
                plt.xlabel('query', fontsize=15)
                plt.legend()
                plt.gca().set_ylim(top=1)
                plt_idx += 1

    plt.tight_layout(rect=(-0.015, -0.02, 1.015, 1.02))
    plt.savefig('result_plots/mono_prior+refpoints_{}'.format(num_iter))
    plt.show()
