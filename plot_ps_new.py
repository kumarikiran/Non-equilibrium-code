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
v = np.array((0.1,0.25, 1.0))

# v = v *L
# Define the power-law function
from scipy.optimize import curve_fit
def power_law(x, a, b):
    return a * x**b
df_para = pd.DataFrame(np.zeros((4,4)))
# Create subplots with 4 columns and 1 row
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # Adjust figsize as needed
# Adjust spacing between columns (optional)
plt.subplots_adjust(wspace=0.4) 
x_limits = (0, 1000)
y_limits = (0, 1000)
for v_i, v_value  in enumerate(v): # parameter loop
    # for rc_i, rc_value in enumerate(rc): # parameter loop 
        cont = np.zeros((L,L))
        data =  pd.read_csv('data_cp/data_orc_mid/cp_data_v%.6f_rc%.6f.csv' %(v_value, rc_value))     
        ax=axes[v_i]
        ############# for the ps vs  s
        y_limits = (0.8, 10000)
        x_limits = (10, 1000)
        ps = np.zeros(L)
        for i in range (1,L):
            ps[i-1] = np.diag(data,i).mean()
        ax.plot(ps,'c--', marker='o', linewidth= 1)    
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ##### Plot the fittings 
        cut_data = int(ps.shape[0]*0.25)
        x_data = np.arange(12, cut_data )
        y_data = ps[12:cut_data]
        
        params, covariance = curve_fit(power_law, x_data, y_data) 
        a_fit, b_fit = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = power_law(x_fit, a_fit, b_fit)
        
        # ax.scatter(x_data, y_data, label='Data')
        ax.plot(x_fit, y_fit, label=f'Fit: a={a_fit:.2f}, b={b_fit:.2f}', color='black', linestyle='--', linewidth=2.5)
        ax.text(0.5, 0.9, f'slope={b_fit:.2f}', transform=ax.transAxes, color='black', fontsize = 18)




for ax in axes:
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.set_xlim(x_limits)

    ax.set_ylim(y_limits)
    ax.set_xlabel('s', fontsize=28)
    ax.set_ylabel('P(s)', fontsize=28)
plt.subplots_adjust(wspace=0.4)            
        
# sns.heatmap(cont/cont.max().max())
                
   # Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.savefig('aa_plot_ps.pdf')

# plt.savefig('uniform_500.png')
plt.show()             




            
        
    
