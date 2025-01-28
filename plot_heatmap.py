
import numpy as np
import pandas as pd 
import seaborn as sns 
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from function_contact import contact_ps
# from function_contact import contact_ps_start

L = 1000
nu = -0.7
rc = np.array((10, 5, 1))  
v = np.array((0.1,0.25, 0.5, 1))  
fig, axes = plt.subplots(v.shape[0], rc.shape[0], 
                         figsize=(rc.shape[0]*5, v.shape[0]*5), 
                         sharex=True, 
                         sharey=True)


for v_i, v_value  in enumerate(v): # parameter loop
    for rc_i, rc_value in enumerate(rc): # parameter loop 
        cont = np.zeros((L,L))
        data =  pd.read_csv('cp_data_v%.6f_rc%.6f.csv' %(v_value, rc_value))
        
        ax=axes[v_i, rc_i]
        ax.set_title(f'v={v_value}, rc={rc_value}(s)^({nu})', fontsize = 24)  
        sns.heatmap(data, ax=axes[v_i, rc_i], cmap="Reds",cbar=True,
                    vmin= 0, vmax=1000)
        
        # Draw the heatmap
        # sns.heatmap(data, ax=ax, cmap="Reds", cbar=v_i == len(v_value) - 1 and rc_i == len(rc_value) - 1,
        #             cbar_ax=None if not (v_i == len(v_value) - 1 and rc_i == len(rc_value) - 1) else cbar_ax,
        #             vmin=0.0001, vmax=0.1)
        
        ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust tick label fontsize
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)  # Adjust x-axis label fontsize
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)                    
   # Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.savefig('cont_L%d_nu%.2f_1.png' %(L, nu))
plt.show()      
