"""
All the specifications we need for the experiment.
"""
import os

SEED = 666

ROOT = os.path.dirname(__file__)

DIR_STORAGE = os.path.join(ROOT, 'storage')

PATH_TRAFFIC_FULL = os.path.join(DIR_STORAGE, 'traffic_full.npy')
PATH_TRAFFIC_PRUNED = os.path.join(DIR_STORAGE, 'traffic_pruned.npy')
PATH_TRAFFIC_MIN = os.path.join(DIR_STORAGE, 'traffic_min.npy')
PATH_TRAFFIC_MAX = os.path.join(DIR_STORAGE, 'traffic_max.npy')

NAMES_OBJ_DELAY = ['D - Car eastbound', 'D - Car westbound', 'D - PT Overtoom out', 'D - PT Overtoom in', 'D - PT Hobbemastraat out', 'D - PT Hobbemastraat in', 'D - Bicycles to Vondelpark']
NAMES_OBJ_QUEUE = ['Q - Block West 1', 'Q - Block West 2', 'Q - Block E1', 'Q - Block E2']

ABBREV_OBJ_DELAY = ['CarE', 'CarW', 'PTO_o', 'PTO_i', 'PTH_o', 'PTH_i', 'BicV']
ABBREV_OBJ_QUEUE = ['BW1', 'BW2', 'BE1', 'BE2']

NUM_OBJECTIVES = len(NAMES_OBJ_DELAY) + len(NAMES_OBJ_QUEUE)
