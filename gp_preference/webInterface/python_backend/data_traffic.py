import numpy as np
import sys
sys.path.insert(0, '../')
from python_backend import specs_traffic


def get_traffic_min():
    return np.load(specs_traffic.PATH_TRAFFIC_MIN)


def get_traffic_max():
    return np.load(specs_traffic.PATH_TRAFFIC_MAX)


def get_traffic_results():
    # results = np.load(specs_traffic.PATH_TRAFFIC_FULL)
    results = np.load(specs_traffic.PATH_TRAFFIC_PRUNED)
    results = np.round(results, 1)
    return results


def get_example_traffic():
    traffic_results = get_traffic_results()
    traffic_avg = np.round(np.average(traffic_results, axis=0), 1)
    return traffic_avg


def get_objective_names():
    return np.concatenate((specs_traffic.NAMES_OBJ_DELAY, specs_traffic.NAMES_OBJ_QUEUE))


def get_objective_abbrev():
    return np.concatenate((specs_traffic.ABBREV_OBJ_DELAY, specs_traffic.ABBREV_OBJ_QUEUE))


if __name__ == '__main__':

    traffic_results = get_traffic_results()

    print(traffic_results)

    traffic_example = get_example_traffic()

    print(traffic_example)

    objective_names = get_objective_names()

    print(objective_names)
