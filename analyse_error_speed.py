import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import linalg as LA

sns.set(style="whitegrid")
plt.figure()
V = []
df = []
for j in range(3,5):
    for i in range(9):
        measurement= pd.read_csv('./output/fly_arene_'+str(j)+'_num_'+ str(i)+'.csv',delimiter=':')
        #print(measurement.head())
        print('max DT: ',max(measurement['Date(ms)'].diff().shift(-1)))
        print('min DT: ',min(measurement['Date(ms)'].diff().shift(-1)))

        vx = measurement['VX (in pixel/s)']
        vy = measurement['VY (in pixel/s)']
        v = LA.norm([[vx,vy]],axis=1)
        #print(j,i,np.where(v>5))
        V = np.append(V,v)
        #print(V)
        print('median : ', measurement.describe())
        df = [df,measurement]
# Draw a violinplot with a narrower bandwidth than the default

print(df)
sns.violinplot(data=V, palette="Set3", bw=.2, cut=1, linewidth=1)

# Finalize the figure
sns.despine(left=True, bottom=True)

plt.show()

plt.figure()
# Draw a nested boxplot to show bills by day and time
sns.boxplot(data=V)


plt.show()
