#  Model constants
CONFIG = {
    'BATCH_SIZE': 50,
    'EPOCHS': 200,
    'WEIGHT_DECAY': 5e-5,
    'INITIAL_LR': 3e-4,
    'CHECKPOINT_DIR': './runs',
    'N_PRED': 9,
    'N_HIST': 12,
    'DROPOUT': 0.2,
    # number of possible 5 minute measurements per day
    'N_DAY_SLOT': 288,
    # number of days worth of data in the dataset
    'N_DAYS': 44,
    # If false, use GCN paper weight matrix, if true, use GAT paper weight matrix
    'USE_GAT_WEIGHTS': True,
    'N_NODE': 228,
}

import numpy as np
#
# RESIDENTIAL_LOAD_FACTOR = np.array([
#     0.80, 0.73, 0.69, 0.64, 0.62, 0.61, 0.60, 0.61, 0.68, 0.77, 0.81, 0.83,
#     0.83, 0.84, 0.86, 0.84, 0.85, 0.84, 0.83, 0.81, 0.98, 1.00, 0.97, 0.90,
# ])
#
# INDUSTRIAL_LOAD_FACTOR = np.array([
#     0.30, 0.28, 0.24, 0.21, 0.20, 0.23, 0.30, 0.50, 0.54, 0.56, 0.58, 0.60, 
#     0.43, 0.40, 0.42, 0.80, 0.87, 0.96, 1.00, 0.97, 0.80, 0.53, 0.38, 0.34,
# ])
#
# COMMERCIAL_LOAD_FACTOR = np.array([
#     0.40, 0.38, 0.34, 0.32, 0.36, 0.47, 0.63, 0.84, 0.94, 1.00, 0.97, 0.88,
#     0.82, 0.60, 0.58, 0.56, 0.53, 0.52, 0.51, 0.48, 0.44, 0.49, 0.43, 0.42,
# ])
#

RESIDENTIAL_LOAD_FACTOR = np.array([
    0.30, 0.40, 0.44, 0.46, 0.50, 0.70, 0.72, 0.80, 0.70, 0.63, 0.50, 0.48,
    0.43, 0.50, 0.44, 0.55, 0.70, 0.85, 1.00, 0.85, 0.75, 0.65, 0.50, 0.44,
])

INDUSTRIAL_LOAD_FACTOR = np.array([
    0.65, 0.60, 0.65, 0.70, 0.80, 0.65, 0.65, 0.60, 0.60, 0.55, 0.50, 0.50, 
    0.50, 0.55, 0.60, 0.65, 0.60, 0.55, 0.68, 0.87, 0.90, 1.00, 0.90, 0.70,
])

COMMERCIAL_LOAD_FACTOR = np.array([
    0.40, 0.38, 0.34, 0.32, 0.36, 0.47, 0.63, 0.84, 0.94, 1.00, 0.97, 0.88,
    0.82, 0.80, 0.72, 0.73, 0.75, 0.65, 0.60, 0.52, 0.44, 0.49, 0.43, 0.42,
])

# SEASON FACTORS
WINTER_LOAD_FACTOR = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
])

SUMMER_LOAD_FACTOR = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
])

AUTUMN_LOAD_FACTOR = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
])

SPRING_LOAD_FACTOR = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
])

# ## Binary vector for loads positioning in the network
RESIDENTIAL_NODES = np.array([
    0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
    1, 1, 1, 0, 1, 0, 0, 1, 0, 0,
    0, 1, 1, 0, 0, 1, 0, 0, 0, 1,
    1, 0, 1
])
COMMERCIAL_NODES = np.array([
    0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
    0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
    0, 1, 0
])
INDUSTRIAL_NODES = np.array([
    1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
    0, 0, 0
])

PV_NODES = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0
])


# class mapping

CLASS_MAPPING = {
    "slack": 0,
    "residential": 1,
    "commercial": 2,
    "industrial": 3,
}
# nodes with strings
#### REMOVED SLACK BUS: THERE ARE ONLY 32 BUSES IN THIS ARRAY
NODE_TYPE = np.array([
    'slack', 'residential',  'residential', 'residential', 'commercial', 'residential', 'industrial', 'industrial', 'industrial', 'commercial',
    'residential', 'residential', 'residential', 'commercial', 'residential', 'industrial', 'industrial', 'residential', 'commercial', 'commercial',
    'residential', 'residential', 'industrial', 'commercial', 'residential', 'industrial', 'commercial', 'industrial', 'commercial', 'residential',
    'residential', 'commercial', 'residential'
])

PEAK_LOAD = np.array([
        0.00, 0.10, 0.09, 0.12, 0.06, 0.06, 0.20, 0.20, 0.06, 0.06, 
        0.045, 0.06, 0.06, 0.12, 0.06, 0.06, 0.06, 0.09, 0.09,  0.09,
        0.09, 0.09, 0.09, 0.42, 0.42, 0.06, 0.06, 0.06, 0.12, 0.20,
        0.15, 0.21, 0.06,
]) 
