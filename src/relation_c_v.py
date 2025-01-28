

import numpy as np
import pandas as pd 
import seaborn as sns 
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

### parameters 
v=20
rc=0.1
L = 1000

rc = np.array((0.2, 0.8))
rc_value = 1
v = np.array((0.1,0.25, 0.5, 1.0))


folder_name= ['mid', 'start', 'both', 'hind_mid']
file_name= ['', 'start_', 'both_', 'hind_']

df = pd.DataFrame(np.zeros((4, 4)), columns=folder_name)
for ii in range (4):
    folder = folder_name[ii]
    file = file_name[ii]
    for v_i, v_value  in enumerate(v): # parameter loop
            cont = np.zeros((L,L))
            data =  pd.read_csv('data_cp/data_orc_%s/%scp_data_v%.6f_rc%.6f.csv' %(folder, file,  v_value, rc_value), index_col=0)  
            
            df.iloc[v_i][folder] = data.sum().sum()
# df = df/(1000*1000)  
# df['mid'] =    df['mid']/df['mid'][0]    
# df['start'] =    df['start']/df['start'][0]  
# df['both'] =    df['both']/df['both'][0]  
# df['hind_mid'] =    df['hind_mid']/df['hind_mid'][0]  
df['index'] = v

cmap = plt.get_cmap('Blues')
blue_shades = [cmap(i/4) for i in range(5)] 

fig, ax = plt.subplots(1, 1, 
                         figsize=(5,5), 
                         sharex=True, 
                         sharey=True)
# ax.set_ylim((20000000, 2000000000))
ax.set_xlim((0.09, 1.07))
plt.yscale('log')
plt.xscale('log')
# ax.plot(df['index'], df['mid'], 'd--', markersize=10, linewidth=1, color='darkred', alpha=1, 
#         label='mid')

# ax.plot(df['index'],df['start'], '^--', markersize=10, linewidth=1,  color = 'lightcoral',  
#         label='start')
# ax.plot(df['index'], df['both'],'>--', markersize=10, linewidth=1,  color = 'red', 
#         label='both')
# ax.plot(df['index'],df['hind_mid'], 'o--', markersize=10, linewidth=1, color='lightcoral', alpha=1, 
#         label='hind')


ax.plot(df['index'], df['mid']/df['mid'][0], 'd--', markersize=14, linewidth=2.5, color=blue_shades[1], alpha=1, 
        markeredgecolor='black', label='mid')

ax.plot(df['index'], df['both']/df['both'][0], '>--', markersize=12, linewidth=2, color=blue_shades[2], 
        markeredgecolor='black', label='both')

ax.plot(df['index'], df['start']/df['start'][0], 'o--', markersize=10, linewidth=1.5, color=blue_shades[3],  
        markeredgecolor='black', label='start')

ax.plot(df['index'], df['hind_mid']/df['hind_mid'][0], '*--', markersize=11, linewidth=1, color=blue_shades[4], alpha=1, 
        markeredgecolor='black', label='pause')


def power_law_fit(x, k, m):
    return k * (x ** m)

# Fit the curve
from scipy.optimize import curve_fit
popt, pcov = curve_fit(power_law_fit, df['index'], df['hind_mid']/df['hind_mid'][0])
k = popt[0]  # The scaling factor
m = popt[1]

# Change the value of k
new_k = k - 0.02 # Set this to your desired value
popt[0] = new_k  


# Generate x values for plotting the fitted curve
x_fit = np.linspace(0.2, 0.7, 100)
y_fit = power_law_fit(x_fit, *popt)

# Fitted exponential curve plot
ax.plot(x_fit, y_fit, color='red', linewidth=2)





ax.set_ylabel(r"$C_t/C_0$" , fontsize=31)
# ax.set_ylabel(r"$C_t$" , fontsize=31)
ax.set_xlabel(r"$v$", fontsize=28)
ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

ax.set_xticks([0.1, 0.25, 0.5, 1])
ax.set_xticklabels([0.1, 0.25, 0.5, 1])
# ax.set_ylabel(r'$p(R_{P, E%s})$' %(num_enhancer), fontsize=31)
ax.legend(loc = 'upper right',  fontsize = 10)
ax.spines['top'].set_linewidth(2)      # Top border
ax.spines['bottom'].set_linewidth(2)   # Bottom border
ax.spines['left'].set_linewidth(2)     # Left border
ax.spines['right'].set_linewidth(2)    # Right border

### distance from the gene 

plotname = 'plot.pdf' 
plt.savefig(plotname, transparent = False, bbox_inches='tight')
plt.show()

