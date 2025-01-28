#!/usr/bin/env python3
"""
This contain the function called by the plot file 
"""

import random 
import numpy as np
def contact_ps(L, v, rc, nu):
    cont = np.zeros((L,L))
    v_pause = 1
    if v < 1:
        v_pause = int(1/v)
        v = 1
    for ii in range (L//2 +5, L, int(v)):  ## time loop 
        for _ in range(v_pause):
            st = L -ii
            ed = ii
            #print(st, ed)
            ### loop for the contact 
            for i in range (st, ed):
                # for j in range (st, i-1):
                for j in range (i+1, ed):
                    random_number = random.random()
                    rc_rule = rc* abs(i -j)**(nu)
                    if random_number < rc_rule:
                        cont[i,j] = cont[i,j] +1
                        cont[j,i] = cont[i,j]
    data = cont/cont.max().max() 
    ps = np.zeros(L)
    for i in range (1,L):
        ps[i-1] = np.diag(data,i).mean()    
    return(cont, ps)


def contact_ps_start(L, v, rc, nu):
    cont = np.zeros((L,L))
    v_pause = 1
    if v < 1:
        v_pause = int(1/v)
        v = 1
    for ii in range (5, L, int(v)):  ## time loop 
    # for ii in range (L//2 +5, L, int(v)):  ## time loop
        for _ in range(v_pause):
            st = 0
            ed = ii
            #print(st, ed)
            ### loop for the contact 
            for i in range (st, ed):
                # for j in range (st, i-1):
                for j in range (i+1, ed):
                    random_number = random.random()
                    rc_rule = rc* abs(i -j)**(nu)
                    if random_number < rc_rule:
                        cont[i,j] = cont[i,j] +1
                        cont[j,i] = cont[i,j]
    data = cont/cont.max().max() 
    ps = np.zeros(L)
    for i in range (1,L):
        ps[i-1] = np.diag(data,i).mean()    
    return(cont, ps)

def contact_ps_start_hind(L, v, rc, nu):
    cont = np.zeros((L,L))
    v_pause = 1
    if v < 1:
        v_pause = int(1/v)
        v = 1
    for ii in range (5, L, int(v)):  ## time loop 
    # for ii in range (L//2 +5, L, int(v)):  ## time loop
        for _ in range(v_pause):
            st = 0
            ed= ii
            #print(st, ed)
            ### loop for the contact 
            for i in range (st, ed):
                # for j in range (st, i-1):
                for j in range (i+1, ed):
                    random_number = random.random()
                    rc_rule = rc* abs(i -j)**(nu)
                    if random_number < rc_rule:
                        cont[i,j] = cont[i,j] +1
                        cont[j,i] = cont[i,j]
                                
            if ii ==250:
                for _ in range(200):
                    for i in range (st, ed):
                        # for j in range (st, i-1):
                        for j in range (i+1, ed):
                            random_number = random.random()
                            rc_rule = rc* abs(i -j)**(nu)
                            if random_number < rc_rule:
                                cont[i,j] = cont[i,j] +1
                                cont[j,i] = cont[i,j]
                                
                                        
                                
    data = cont/cont.max().max() 
    ps = np.zeros(L)
    for i in range (1,L):
        ps[i-1] = np.diag(data,i).mean()    
    return(cont, ps)


def contact_ps_both(L, v, rc, nu):
    cont = np.zeros((L,L))
    v_pause = 1
    if v < 1:
        v_pause = int(1/v)
        v = 1
    for ii in range (1, int(2*L/3), int(v)):  ## time loop 
    # for ii in range (L//2 +5, L, int(v)):  ## time loop
        for _ in range(v_pause):
            st = 0
            ed = ii
            #print(st, ed)
            ### loop for the contact 
            for i in range (st, ed):
                # for j in range (st, i-1):
                for j in range (i+1, ed):
                    random_number = random.random()
                    rc_rule = rc* abs(i -j)**(nu)
                    if random_number < rc_rule:
                        cont[i,j] = cont[i,j] +1
                        cont[j,i] = cont[i,j]
                    
        for _ in range(v_pause):
            st = L-ii
            ed = L
            #print(st, ed)
            ### loop for the contact 
            for i in range (st, ed):
                # for j in range (st, i-1):
                for j in range (i+1, ed):
                    random_number = random.random()
                    rc_rule = rc* abs(i -j)**(nu)
                    if random_number < rc_rule:
                        cont[i,j] = cont[i,j] +1
                        cont[j,i] = cont[i,j]
    data = cont/cont.max().max() 
    ps = np.zeros(L)
    for i in range (1,L):
        ps[i-1] = np.diag(data,i).mean()    
    return(cont, ps)


