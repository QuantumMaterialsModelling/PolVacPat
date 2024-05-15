import numpy as np
import pickle
from configurational_ML.sim_anneal import sim_anneal, generate_random_config
import sys

seed, store = int(sys.argv[1]), sys.argv[2]

print(seed, store)

with open('params/weights_combined_cells.pkl', 'rb') as f:
    weights = pickle.load(f)

with open('dataset/processed_dataset_combined.pkl', 'rb') as f:
    _, _, _, desc_scaler, _ = pickle.load(f)


# temperatures in internal units as used by the ML model
temps = np.concatenate([np.ones(40000)*1.6e-5, np.ones(10000)*1e-5])

configuration = generate_random_config((24,16), 1.0, seed=seed)

# anneals both polarons and VOs
confs,ens = sim_anneal(configuration, weights, desc_scaler, temps=temps, seed=seed, restrict_ov=False, random=0.0)

with open(store, 'wb') as f:
    pickle.dump([confs, ens], f)
