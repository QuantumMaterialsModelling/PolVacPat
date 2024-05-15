from configurational_ML.dataset import extract_data, augment, convert_to_desc
import numpy as np
import pickle
import sys

path_store, width_x, width_y, alpha = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])

cell = np.diag(np.loadtxt('dataset/POSCAR_6x4',skiprows=2, max_rows=3))

print('-'*50)
print('Extracting data:')
configurations, energies = extract_data('dataset/data_6x4.pkl', cell, partitions=(12,8))
print(len(configurations), ' extracted')

print('-'*50)
print('Augmenting data:')
configurations, energies = augment(configurations, energies)
print(len(configurations), ' configurations after augmentation')

print('-'*50)
print('Calculating Descriptor:')
X, pol_type, Y, desc_scaler, en_scaler = convert_to_desc(configurations, energies, size = (width_x, width_y), alpha=alpha)

print('-'*50)
print('Storing Data')
with open(path_store, 'wb') as f:
    pickle.dump([X, pol_type, Y, desc_scaler, en_scaler], f)
