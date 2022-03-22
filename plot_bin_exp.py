import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_pickle('MIPLIB_RESULTS/Testing/Bin_exp/Random_problem_multi_index.pkl')
print(df)

x = df.index + np.ones_like(df.index)
y_normal = df[('Normal','Time')]
y_opt = df[('Optimized','Time')]

plt.plot(x,y_normal,label='Normal')
plt.plot(x,y_opt,label='Optimized')
plt.ylabel('Runtimes')
plt.xlabel('Upper Bound')
plt.legend()
plt.show()