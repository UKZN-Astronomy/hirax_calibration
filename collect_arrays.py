import numpy as np
import os

''' This is a simple script to collect all of the file outputs from individual CorrCal runs (saved in an `output_runs' directory), and create a 2D numpy array (full_runs.npy)
containing all recovered gains.'''

full_runs_output = np.array([])
data_path = '/data/zahrakad/hirax_corrcal/output_runs/'
aa = [file for file in os.listdir(data_path)]
for data in aa:
    inf_from_every_file = np.load(os.path.join(data_path,data))
    full_runs_output = np.append(full_runs_output, inf_from_every_file)
    full_runs_output = full_runs_output.reshape(-1, len(inf_from_every_file))

np.save('/data/zahrakad/hirax_corrcal/full_runs.npy', full_runs_output)
