# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:11:18 2020

@author: Henrik
"""

## Optimization
# work in progress
from add_func import opti_fun_data

dist = np.zeros((1,Ntrain))#np.zeros((Ntest,Ntrain))
min_dist = np.zeros((Ntest,1))
predictor_dist = np.zeros((Ntest,Nparameters))
for i in range(1):#range(Ntest):
    for j in range(Ntrain):
        dist[i,j]  = np.mean(((y_test[i,:]-y_train[j,:])/y_test[i,:])**2)
    min_dist[i] = np.argmin(dist[i,:])
    predictor_dist[i,:] = X_train[int(min_dist[i][0]),:]
    
    




import functools as functools
from add_func import bsimpvola,HNC_Q
from config import r
"""
def optimization_fun (prediction,x):
    alpha = x[0]
    beta = x[1]
    gamma_star = x[2]
    omega = x[3] 
    h0 = x[4]
    err = 0
    for i in range(Nstrikes):
        st = strikes[i]
        for t in range(Nmaturities):
            mat = maturities[t]
            vola = prediction[t,i]
            err += ((vola-bsimpvola(HNC_Q(alpha, beta, gamma_star, omega, h0, 1, st, r, mat, 1),1,st,mat,r,'c'))/vola)**2
    return err/(Nmaturities*Nstrikes)
"""
from py_vollib.black_scholes.implied_volatility import black_scholes
def optimization_fun_prices (prediction,x):
    alpha = x[0]
    beta = x[1]
    gamma_star = x[2]
    omega = x[3] 
    h0 = x[4]
    err = 0
    for i in range(Nstrikes):
        st = strikes[i]
        for t in range(Nmaturities):
            mat = maturities[t]
            vola = prediction[t,i]
            err += ((black_scholes('c',1,st,mat,r,vola)-HNC_Q(alpha, beta, gamma_star, omega, h0, 1, st, r, mat, 1))\
                    /black_scholes('c',1,st,mat,r,vola))**2
    return err/(Nmaturities*Nstrikes)               


from multiprocessing import Pool
import os

def testfun(prediction,st,t,x,i):
    alpha = x[0]
    beta = x[1]
    gamma_star = x[2]
    omega = x[3] 
    h0 = x[4]
    mat = maturities[t]
    vola = prediction[t,i]
    return ((black_scholes('c',1,st,mat,r,vola)-HNC_Q(alpha, beta, gamma_star, omega, h0, 1, st, r, mat, 1))\
                    /black_scholes('c',1,st,mat,r,vola))**2

def optimization_fun_pricesparallel (prediction,x):
    err = 0
    for i in range(1):#range(Nstrikes):
        st = strikes[i]
        try:
            pool = Pool(np.max([os.cpu_count()-1,1]))
            err += np.sum(pool.map(functools.partial(testfun, prediction, st, t, x), range(Nmaturities)))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join() 
    return err/(Nmaturities*Nstrikes)


x0 = predictor_dist[0,:]
tmp = yinversetransform(prediction[0,:,:])
    
t.start()
optimization_fun_pricesparallel(tmp,x0)
t.stop()



bounds = ([0, 10], [0, 1-(1e-6)],[-1000,1000],[np.finfo(float).eps,10],[0, 10])
ineq_cons = {'type': 'ineq','fun' : lambda x: np.array([1 - x[0]*x[1]**2-x[1]])}
res2 =[]
for n in range(1):#range(Ntest):
    x0 = predictor_dist[n,:]
    tmp = yinversetransform(prediction[n,:,:])
    res2.append(minimize(functools.partial(optimization_fun_prices,tmp), x0, method='SLSQP', constraints=[ineq_cons],options={'disp': 1}, bounds=bounds))

x0 = predictor_dist[0,:]
tmp = yinversetransform(prediction[0,:,:])
    
t.start()
optimization_fun_prices(tmp,x0)
t.stop()

"""def cons_hng(x):
    return x[0]**2*x[2]+x[1]
nonlinear_constraint = NonlinearConstraint(cons_hng, -np.inf, 1)
bounds = ([0, 10], [0, 1-(1e-6)],[-1000,1000],[np.finfo(float).eps,10],[0, 10])
res = minimize(opti_fun_data(prediction), x0, method='trust-constr',
               constraints=[nonlinear_constraint],options={'verbose': 1}, bounds=bounds)
"""

"""
#slsqp
bounds = ([0, 10], [0, 1-(1e-6)],[-1000,1000],[np.finfo(float).eps,10],[0, 10])
ineq_cons = {'type': 'ineq','fun' : lambda x: np.array([1 - x[0]*x[1]**2-x[1]])}
res2 =[]
for n in range(1):#range(Ntest):
    x0 = predictor_dist[n,:]
    res2.append(minimize(opti_fun_data(prediction[n,:,:]), x0, method='SLSQP', constraints=[ineq_cons],\
                options={'ftol': 1e-9, 'disp': True},bounds=bounds))
"""