from configurational_ML.ML import train, init_weights
import pickle
import sys

path_load, path_params, path_store = sys.argv[1], sys.argv[2], sys.argv[3]

with open(path_load, 'rb') as f:
    X, pol_type, Y, desc_scaler, en_scaler = pickle.load(f)

initial_weights = init_weights(3, X.shape[-1], 32, 10)

res = train(X, 
            Y, 
            pol_type, 
            initial_weights, 
            epochs=10000, 
            eta=0.005, 
            batch_size=16,
            store_path=path_params)


with open(path_store, 'wb') as f:
    pickle.dump(res, f)
