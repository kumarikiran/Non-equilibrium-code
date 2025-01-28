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

### parameters 
v=1
rc=5
L = 100
len_comb=10
nu = -0.7

# Define the power-law function
def power_law(x, a, b):
    return a * x**b
df_para = pd.DataFrame(np.zeros((4,4)))
k = 0


comb_data=np.zeros((L*len_comb, L*len_comb))
for comb_i in range(len_comb):
    data, ps =   contact_ps(L, v, rc, nu)
    st = comb_i*L
    ed = comb_i*L +L
    comb_data[st:ed, st:ed] = data

for i in range (comb_data.shape[0]*comb_data.shape[0] ):
    rand_i = int(random.random() * comb_data.shape[0])
    rand_j = int(random.random() * comb_data.shape[0])
    if (rand_i != rand_j):
        comb_data[rand_i, rand_j]  = 5 *( abs(rand_i - rand_j)**(-1.3))
    
        
    
fig, axes = plt.subplots(1, 2, 
                         figsize=(12, 6), 
                         sharex=False, 
                         sharey=False)
        
ax=axes[0]
sns.heatmap(comb_data, ax=axes[0], cmap="Reds",cbar=False,
            vmin= 0.0001, vmax=0.1)
ax.set_title(f'v={v}, rc={rc}(s)^({nu})') 

ax=axes[1]
ps = np.zeros(comb_data.shape[0])
for i in range (1,comb_data.shape[0]):
    ps[i-1] = np.diag(comb_data,i).mean()
ax.plot(ps,'c', marker='o', linewidth= 1) 

cut_data = int(comb_data.shape[0]*0.1)
x_data = np.arange(15, cut_data )
y_data = ps[15:cut_data]

params, covariance = curve_fit(power_law, x_data, y_data) 
a_fit, b_fit = params
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = power_law(x_fit, a_fit, b_fit)

# ax.scatter(x_data, y_data, label='Data')
ax.plot(x_fit, y_fit, label=f'Fit: a={a_fit:.2f}, b={b_fit:.2f}', color='red')
ax.text(0.5, 0.9, f'slope={b_fit:.2f}', transform=ax.transAxes, color='blue')
        
ax.set_ylim([0.0001, 1])
ax.set_xlim([1, comb_data.shape[0]])        
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f'v={v}, rc={rc}(s)^({nu})') 
           
plt.tight_layout()
plt.savefig('comb_plot_L%d_nu%.2f_v%s_rc%s_extra.png' %(L, nu, v, rc))
plt.show()    
            
        
    
