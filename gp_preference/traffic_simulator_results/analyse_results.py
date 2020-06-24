import numpy as np
import pandas as pd
from pymodem import Value, pruners


random_seeds = [5, 21, 42, 333, 11235]

# read in the results
results = [pd.read_csv('./results/results_{}.csv'.format(i)) for i in random_seeds]

# concatenate results to large matrix
results = np.concatenate(results).reshape((-1, results[0].shape[0], results[0].shape[1]))

NAMES_OBJ_DELAY = ['Car eastbound', 'Car westbound', 'PT Overtoom out', 'PT Overtoom in', 'PT Hobbemastraat out', 'PT Hobbemastraat in', 'Bicycles to Vondelpark']
NAMES_OBJ_QUEUE = ['Block West 1', 'Block West 2', 'Block E1', 'Block E2']

NUM_OBJ_DELAY = len(NAMES_OBJ_DELAY)
NUM_OBJ_QUEUE = len(NAMES_OBJ_QUEUE)

NUM_MEASURES_DELAY = 9 * NUM_OBJ_DELAY
NUM_MEASURES_QUEUE = 9 * NUM_OBJ_QUEUE

NUM_ARMS = 256

# split results into delay and queue
results_delay = results[:, :, :NUM_MEASURES_DELAY]
results_queue = results[:, :, NUM_MEASURES_DELAY:]

# only get the results of the objectives that are of interest to us
results_delay = results_delay[:, :, -NUM_OBJ_DELAY:]
results_queue = results_queue[:, :, -NUM_OBJ_QUEUE:]

# get the mean values
results_delay_mean = np.mean(results_delay, axis=0)
results_queue_mean = np.mean(results_queue, axis=0)

# concatenate into one large matrix
results_matrix = -np.hstack((results_delay_mean, results_queue_mean))

np.save('../webInterface/python_backend/storage/traffic_full', results_matrix)

print(results_matrix.shape)

# prune dataset with new datapoint
vv_ccs = Value.ValueVectorSet()
vv_ccs.addAll(results_matrix)
results_matrix_pruned = np.array(pruners.c_prune(vv_ccs).set)

np.save('../webInterface/python_backend/storage/traffic_pruned', results_matrix_pruned)

print(results_matrix_pruned.shape)

# plt.figure()
# plt_height = 3
# plt_width = 4
# for i in range(NUM_OBJ_DELAY+NUM_OBJ_QUEUE):
#     plt.subplot(plt_height, plt_width, i+1)
#     plt.plot(np.sort(results_matrix_pruned[:, i]))
#
# sorting_order = np.argsort(results_matrix_pruned[:, 0])
# for j in range(NUM_OBJ_DELAY+NUM_OBJ_QUEUE):
#     plt.subplot(plt_height, plt_width, i+2)
#     plt.plot(results_matrix_pruned[sorting_order, j])
#
# plt.show()

# normalise the objective values between 0 and 1
rmpn = (results_matrix_pruned - np.min(results_matrix_pruned, axis=0)) / (np.max(results_matrix_pruned, axis=0) - np.min(results_matrix_pruned, axis=0))


# now define some random utility function
obj_weights = np.array(list(range(NUM_OBJ_QUEUE+NUM_OBJ_DELAY)), dtype=np.float) + 1
obj_weights /= np.sum(obj_weights)


def utility_function(datapoints):
    utility = 0
    for i in range(NUM_OBJ_DELAY+NUM_OBJ_QUEUE):
        utility += obj_weights[i] * datapoints[:, i]
    return utility

print(utility_function(rmpn))
