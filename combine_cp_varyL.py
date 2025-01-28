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
# L = np.array([100, 200, 100, 300, 100, 200, 300])
# random_numbers = [np.random.randint(300, 1200) for _ in range(10)]
# L = np.array([1051, 361, 1142, 1131, 712, 911, 436, 375, 1043, 374])
L = np.array([100, 300, 700, 1000])
v = np.array([0.5, 0.1, 2, 2])
L = L*0.8
L = L.astype(int)
len_comb= int(L.sum())
print(len_comb, len(L))
nu = -0.7

# Define the power-law function
def power_law(x, a, b):
    return a * x**b
df_para = pd.DataFrame(np.zeros((4,4)))
k = 0



comb_data=np.zeros((len_comb, len_comb))
print(comb_data.shape)
st = 0
for comb_i in range(len(L)):
    v_i = v[comb_i]
    data, ps =   contact_ps(L[comb_i], v_i, rc, nu)
    ed = st + L[comb_i]
    print(st, ed, data.shape)
    comb_data[st:ed, st:ed] = data
    st = ed
comb_data = comb_data/comb_data.max()
rc1 = 1
for i in range (comb_data.shape[0]*comb_data.shape[0] ):
    rand_i = int(random.random() * comb_data.shape[0])
    rand_j = int(random.random() * comb_data.shape[0])
    if (rand_i != rand_j):
        comb_data[rand_i, rand_j]  = comb_data[rand_i, rand_j] + (rc1 *( abs(rand_i - rand_j)**(-1.3)))
    
        
from matplotlib.colors import LogNorm, Normalize  
fig, ax = plt.subplots(1, 1, 
                         figsize=(6, 6), 
                         sharex=False, 
                         sharey=False)
        

# sns.heatmap(comb_data, ax=ax, cmap="Reds",cbar=False,
#             vmin= 0.0001, vmax=0.1)
# sns.heatmap(comb_data, ax=ax, cmap="Reds", cbar=False, vmin=0.0001, vmax=0.1)
color_map = plt.cm.get_cmap('Reds')
# heatmap = ax.imshow(comb_data, cmap=color_map, interpolation='nearest',norm=LogNorm(vmin=1, vmax=1100))
heatmap = ax.imshow(comb_data, cmap=color_map, interpolation='nearest',norm=LogNorm(vmin=0.001, vmax=5))
# Set the number of ticks
# ax.tick_params(axis='both', which='major', labelsize=18) 
        

           
plt.tight_layout()
plt.savefig('comb_plot.png')
plt.show()    
            
        
    
