

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
rc_value = 5
v = np.array((0.1,0.25, 1.0))
# v = np.array((0.1,1, 10))
# v = v *L

# Create subplots with 4 columns and 1 row
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # Adjust figsize as needed
# Adjust spacing between columns (optional)
plt.subplots_adjust(wspace=0.4) 
x_limits = (0, 1000)
y_limits = (0, 1000)
for v_i, v_value  in enumerate(v): # parameter loop
    # for rc_i, rc_value in enumerate(rc): # parameter loop 
        cont = np.zeros((L,L))
        data =  pd.read_csv('data_cp/data_orc_mid/cp_data_v%.6f_rc%.6f.csv' %(v_value, rc_value), index_col=0)   
        
        ### make a reference matrix 
        nu = 0.7
        import random 
        data_ref = np.zeros((1000, 1000), dtype = float)
        for i in range (0, 1000):
            for j in range (i+1, 1000):
                # random_number = random.random()
                # rc_rule = rc_value* abs(i -j)**(nu)
                # if random_number < rc_rule:
                data_ref[i,j] = data_ref[i,j] +abs(i -j)**(nu)
                data_ref[j,i] = data_ref[i,j]
        data_ref = data_ref/data_ref.max().max() 
                    
                    
        ax=axes[v_i]
        color_map = plt.cm.get_cmap('Reds')
        heatmap = ax.imshow(data, cmap=color_map, interpolation='nearest',norm=LogNorm(vmin=1, vmax=500))
        
        cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        # v_value = int(v_value)
        ax.set_title(f'$v={v_value}$', fontsize = 24)    
        ax.tick_params(axis='x', labelsize=24) 
        
        ############## for the ps vs  s
        # y_limits = (0.8, 5000)
        # x_limits = (8, 1000)
        # ps = np.zeros(L)
        # for i in range (1,L):
        #     ps[i-1] = np.diag(data,i).mean()
        # ax.plot(ps,'c--', marker='o', linewidth= 1)    
        # ax.set_xscale('log')
        # ax.set_yscale('log')
for ax in axes:
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    # ax.set_xlabel('s', fontsize=24)
    # ax.set_ylabel('P(s)', fontsize=24)

        
plt.subplots_adjust(wspace=0.4)            
        
# sns.heatmap(cont/cont.max().max())
                
   # Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.savefig('aa_plot.pdf')

# plt.savefig('uniform_500.png')
plt.show()             




            
        
    
