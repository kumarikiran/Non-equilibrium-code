#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is for Uniform velocity 
This takes the 5 value of the parameter v and rc and 
generates the contact map for each case
"""

import numpy as np
import pandas as pd 
import seaborn as sns 
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

### parameters 
v=20
rc=1
rc_value = rc
L = 1000

rc = np.array((0.2, 0.8))

v = np.array((0.1,  0.25 , 0.5, 1, 10))

# Create subplots with 4 columns and 1 row
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))  # Adjust figsize as needed
# Adjust spacing between columns (optional)
plt.subplots_adjust(wspace=0.4) 
x_limits = (0, L)
y_limits = (0, L)
for v_i, v_value  in enumerate(v): # parameter loop
    # for rc_i, rc_value in enumerate(rc): # parameter loop 
        cont = np.zeros((L,L))
        v_write = v_value
        if v_value < 1:
            v_pause = int(1/v_value)
            v_value = 1
        for ii in range (L//2 +5, L, int(v_value)):  ## time loop 
            for _ in range(v_pause):
                st = L -ii
                ed = ii
                #print(st, ed)
                ### loop for the contact 
                for i in range (st, ed):
                    # for j in range (st, i-1):
                    for j in range (i+1, ed):
                        random_number = random.random()
                        if random_number < rc_value:
                            cont[i,j] = cont[i,j] +1
                            cont[j,i] = cont[i,j]
        data = cont #/cont.max().max()           ## for heatmap    
        pd.DataFrame(data).to_csv('uniform_data_v%f_rc%f.csv' %(v_write, rc_value))
        # ax=axes[v_i]
        color_map = plt.cm.get_cmap('Reds')
        # heatmap = ax.imshow(data, cmap=color_map, interpolation='nearest', norm=LogNorm(), vmin=1, vmax=100)
        # data = pd.DataFrame(data)
        # heatmap = ax.imshow(data, cmap=color_map, interpolation='nearest',norm=LogNorm(vmin=1, vmax=100))
        # cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.tick_params(labelsize=14)
        # pd.DataFrame(data).to_csv('data_cp/data_uniform/uniform_v%d\f_rc%d.csv' %(v_i, rc_i))
        # axes[v_i, rc_i].set_title(f'v={v_value}, rc={rc_value}', fontsize = 24)    
        # axes[v_i, rc_i].tick_params(axis='x', labelsize=24) 
        
        ############## for the ps vs  s
        # ps = np.zeros(L)
        # for i in range (1,L):
        #     ps[i-1] = np.diag(data,i).mean()
        # ax.plot(ps,'c--', marker='o', linewidth= 1)    
        # ax.set_xscale('log')
        # ax.set_yscale('log')
# for ax in axes:
#     ax.tick_params(axis='y', labelsize=16)
#     ax.tick_params(axis='x', labelsize=16)
#     ax.spines['top'].set_linewidth(1)
#     ax.spines['bottom'].set_linewidth(1)
#     ax.spines['left'].set_linewidth(1)
#     ax.spines['right'].set_linewidth(1)
#     ax.set_xlim(x_limits)
#     y_limits = (0.8, 1000)
#     ax.set_ylim(y_limits)
#     ax.set_xlabel('s', fontsize=24)
#     ax.set_ylabel('P(s)', fontsize=24)

        
plt.subplots_adjust(wspace=0.4)            
        
# sns.heatmap(cont/cont.max().max())
                
   # Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.savefig('aa_plot_ps.pdf')

# plt.savefig('uniform_500.png')
plt.show()             




            
        
    
