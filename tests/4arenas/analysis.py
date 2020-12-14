#%% Imports
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# %% Loading dataframe from new result
df_new = pd.read_hdf('df_batch_0.hdf', key='df')
# %% Loading dataframe from old result
df_old = pd.read_hdf('../../data/testing_data/4arenas/old_combined.hdf', 'data')

# %% We need to match the right frame numbers
initial_frame = np.max([df_old['frame'].min(), df_new['frame'].min()])
final_frame = np.min([df_old['frame'].max(), df_new['frame'].max()])

df_old = df_old.query(f'{initial_frame} <= frame <= {final_frame}')[['frame', 'ID', 'x', 'y']]
df_new = df_new.query(f'{initial_frame} <= frame <= {final_frame}')


# %% Calculating distances
n_flies_old = df_old.ID.unique().size
n_flies_new = df_new.ID.unique().size 
distances = ((df_new[['x', 'y']].to_numpy().reshape(-1, n_flies_new, 1, 2) - df_old[['x', 'y']].to_numpy().reshape(-1, 1, n_flies_old, 2))**2).sum(axis=-1)


# %% Matching flies
match = linear_sum_assignment(distances[:, :, 0])
# %% Check where the maximum distance is larger than threshold
threshold = 50
danger_idx = np.nonzero(distances[match[0], match[1], 2] > threshold)[0]

# %% Plotting the two trajectories
idx = danger_idx[0]
plt.plot(df_old.query(f'ID=={match[1][idx]}').x, df_old.query(f'ID=={match[1][idx]}').y)
plt.plot(df_new.query(f'ID=={match[0][idx]}').x, df_new.query(f'ID=={match[0][idx]}').y)
# %% Plotting the distance as function of time
def trajectory_distance(trajectory_1, trajectory_2):
    """    Calculates mean and max distance between two trajectories. 
    Subtracts initial position. 
    """
    # We start both trajectories at zero
    trajectory_1, trajectory_2 = trajectory_1.to_numpy(), trajectory_2.to_numpy()
    trajectory_1 -= trajectory_1[0, :]
    trajectory_2 -= trajectory_2[0, :]

    distance = np.sqrt((trajectory_1**2 - trajectory_2**2).sum(axis=1))
    return distance


idx = 5#danger_idx[0]
distance = trajectory_distance(df_new.query(f'ID == {match[0][idx]}')[['x', 'y']], df_old.query(f'ID == {match[1][idx]}')[['x', 'y']])
plt.plot(distance)


# %%
