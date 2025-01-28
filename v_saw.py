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
L = 1000

rc = np.array((1, 25))  
v = np.array((1, 100))   # 
nu = -1.0
para= v/rc

# Define the power-law function
def power_law(x, a, b):
    return a * x**b
df_para = pd.DataFrame(np.zeros((2,2)))
k = 0

fig, axes = plt.subplots(rc.shape[0], v.shape[0], 
                         figsize=(10, 10), 
                         sharex=True, 
                         sharey=True)

for v_i, v_value  in enumerate(v): # parameter loop
    for rc_i, rc_value in enumerate(rc): # parameter loop 
        cont = np.zeros((L,L))
        data, ps =   contact_ps_start(L, v_value, rc_value, nu)
        
        ax=axes[v_i, rc_i]
        ax.set_title(f'v={v_value}, rc={rc_value}(s)^({nu})', fontsize = 24)  
        sns.heatmap(data, ax=axes[v_i, rc_i], cmap="Reds",cbar=False,
                    vmin= 0.0001, vmax=0.1)

                     
   # Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.savefig('start_saw_cp_L%d_nu%.2f.png' %(L, nu))
plt.show()             

fig2, axes = plt.subplots(rc.shape[0], v.shape[0], 
                         figsize=(12, 12), 
                         sharex=True, 
                         sharey=True)
for v_i, v_value  in enumerate(v): # parameter loop
    for rc_i, rc_value in enumerate(rc): # parameter loop 
        cont = np.zeros((L,L))
        data, ps =   contact_ps_start(L, v_value, rc_value, nu)
        
        ax=axes[v_i, rc_i]
        ax.set_title(f'v={v_value}, rc={rc_value}(s)^({nu})', fontsize = 24)  
        ax.plot(ps,'c', marker='o', linewidth= 1)      
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        cut_data = int(L*0.8)
        x_data = np.arange(4, cut_data )
        y_data = ps[4:cut_data]
        
        params, covariance = curve_fit(power_law, x_data, y_data) 
        a_fit, b_fit = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = power_law(x_fit, a_fit, b_fit)
        
        df_para.iloc[rc_i][v_i] = b_fit  
        ax.scatter(x_data, y_data, label='Data')
        ax.plot(x_fit, y_fit, label=f'Fit: a={a_fit:.2f}, b={b_fit:.2f}', color='red')
        ax.text(0.5, 0.9, f'slope={b_fit:.2f}', transform=ax.transAxes, color='blue')

        ### save the ps value 
        df_ps = pd.DataFrame(ps)
        # df_ps.to_csv('data/ps_saw_v%d_rc_%.1f_L%d.txt' %(v_value, rc_value, L), 
                      # index = True, sep = '\t')
plt.tight_layout()
plt.savefig('start_saw_ps_L%d_nu%.2f.png' %(L, nu))
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

            
        
    
