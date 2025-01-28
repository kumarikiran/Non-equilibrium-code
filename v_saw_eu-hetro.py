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
from scipy.optimize import curve_fit
from function_contact import contact_ps
from function_contact import contact_ps_start
### parameters 
v=20
rc=0.5
L = 100

rc = np.array((5, 1))  
v = np.array((0.1,0.25, 1))   # 
nu = -0.7
#para= v/rc

# Define the power-law function
def power_law(x, a, b):
    return a * x**b
df_para = pd.DataFrame(np.zeros((2,2)))
para_list = []
k = 0

fig, axes = plt.subplots(v.shape[0], rc.shape[0], 
                         figsize=(rc.shape[0]*5, v.shape[0]*5), 
                         sharex=True, 
                         sharey=True)

for v_i, v_value  in enumerate(v): # parameter loop
    for rc_i, rc_value in enumerate(rc): # parameter loop 
        cont = np.zeros((L,L))
        data, ps =   contact_ps(L, v_value, rc_value, nu)
        
        ax=axes[v_i, rc_i]
        ax.set_title(f'v={v_value}, rc={rc_value}(s)^({nu})', fontsize = 24)  
        sns.heatmap(data, ax=axes[v_i, rc_i], cmap="Reds",cbar=False,
                    vmin= 0.0001, vmax=0.1)
        ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust tick label fontsize
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)  # Adjust x-axis label fontsize
        ax.set_ylabel(ax.get_ylabel(), fontsize=20) 
        para_list.append([v_value, rc_value, data.sum().sum()])
        # pd.DataFrame(data).to_csv('cp_data_v%f_rc%f.csv' %(v_value, rc_value))

pd.DataFrame(para_list).to_csv('v_para_n%s_1.csv' %L)                     
   # Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.savefig('mid_saw_cp_L%d_nu%.2f_1.png' %(L, nu))
plt.show()             




#### heatmap 
# fig, axes = plt.subplots(1, 1, 
#                          figsize=(12, 12), 
#                          sharex=True, 
#                          sharey=True)

# heatmap = sns.heatmap(df_para, annot=True, fmt=".3f")
# # Change x-axis and y-axis labels
# heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
# # heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=10)
# plt.savefig('para_heatmap.png')
# plt.show()             

            
        
    
