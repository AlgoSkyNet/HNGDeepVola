import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import hngoption as hng
#import py_vollib.black_scholes.implied_volatility as vol!
from calcbsimpvol import calcbsimpvol 
def HNG_MC(alpha, beta, gamma, omega, d_lambda, S, K, rate, T, dt, PutCall = 2, num_path = int(1e6), 
           risk_neutral = True, Variance_specs = "unconditional",output="price"):
    """
    This function calculates the Heston-Nandi-GARCH(1,1) option price of european calls/puts with MonteCarloSim
    Requirements: numpy
    Model Parameters (riskfree adjust will be done automatically): 
        alpha,beta,gamma,omega,d_lambda
    Underlying:
        S starting value, K np.array of Strikes, dt Timeshift, T maturity in dt, r riskfree rate in dt
    Function Parameters:
        Putcall: option type (1=call,0=put,2=both)
        num_path: number of sim paths
        risk_neutral: Baysian, type of simulation
        Variance_specs: Type of inital variance or inital variance input
    
    (C) Henrik Brautmeier, Lukas Wuertenberger 2019, University of Konstanz
    """
    
    gamma_star = gamma+d_lambda+0.5
    
    # Variance Input =========================================================
    if Variance_specs=="unconditional":
        V = (omega+alpha)/(1-alpha*gamma**2-beta)
    elif Variance_specs=="uncondtional forecast":
        sigma2=(omega+alpha)/(1-alpha*gamma_star**2-beta)
        V = omega+alpha-2*gamma_star*np.sqrt(sigma2)+(beta+gamma_star**2)*sigma2
    elif any(type(Variance_specs)==s for s in [float,int,type(np.array([0])[0]),type(np.array([0.0])[0])]):
        V = Variance_specs  #checks if Variance_specs is a float,int,numpy.int32 or numpy.float64
    else:
        print("Variance format not recognized. Uncondtional Variance will be used")
        V = (omega+alpha)/(1-alpha*gamma**2-beta) 
    # ========================================================================
    
    # Initialisation
    r = np.exp(rate*dt)-1                             
    lnS = np.zeros((num_path,T+1))
    h = np.zeros((num_path,T+1))
    lnS[:,0] = np.log(S)*np.ones((num_path))
    h[:,0] = V*np.ones((num_path)) #initial wert
    z = np.random.normal(size=(num_path,T+1))
    
    # Simulation
    for t in np.arange(dt,T+dt,dt):
        if risk_neutral:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma_star*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r-0.5*h[:,t]+np.sqrt(h[:,t])*z[:,t]
        else:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r+d_lambda*h[:,t]+np.sqrt(h[:,t])*z[:,t]
    S_T = np.exp(lnS[:,-1])
    
    # Output
    if PutCall==1: # Call
        price = np.exp(-rate*T)*np.mean(np.maximum(S_T[:,np.newaxis] - K,np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
    elif PutCall==0: # Put
        price = np.exp(-rate*T)*np.mean(np.maximum(K-S_T[:,np.newaxis],np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
    elif PutCall==2: # (Call,Put)
        price_call,price_put =  np.exp(-rate*T)*np.mean(np.maximum(S_T[:,np.newaxis] - K,np.zeros((S_T.shape[0],K.shape[0]))),axis=0),np.exp(-rate*T)*np.mean(np.maximum(K-S_T[:,np.newaxis],np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
        price = (price_call,price_put)
    
    if output=="price":
        return price
    elif output=="bsvola":
        if PutCall!=2:
            K_tmp,tau = np.meshgrid(K.reshape((K.shape[0],1)),np.array(T/252))
            return calcbsimpvol(dict(cp=np.asarray(PutCall), P=np.asarray(price.reshape((1,price.shape[0]))), S=np.asarray(S), K=K_tmp, tau=tau, r=np.asarray(rate*252), q=np.asarray(0)))
    


"""
todo + fragen
    simulatious maturity implemenatation         
    black scholes implied vola (teileweise implementiert
    bs input were sind jährlich? unser input is täglich? lösung1: daily convertion *252 (implementiert)
    lösung2=yearly vola output convertieren? (nicht implementiert,geht das überhaupt?)
     



# BS implied vola TEStING LABRATORY        
# beispiel des erstellers    
#für documentation calcbsimpvol öffnen
S = np.asarray(100)         #Spotprice
K_value = np.arange(40, 160, 25)
K = np.ones((np.size(K_value), 1))
K[:, 0] = K_value           #strike
tau_value = np.arange(0.25, 1.01, 0.25)
tau = np.ones((np.size(tau_value), 1))
tau[:, 0] = tau_value       #T-t
r = np.asarray(0.01)        #riskfree
q = np.asarray(0.03)        #dividen yield
cp = np.asarray(1)          #call=1 put =-1
P = [[59.35, 34.41, 10.34, 0.50, 0.01],
[58.71, 33.85, 10.99, 1.36, 0.14],
[58.07, 33.35, 11.50, 2.12, 0.40],
[57.44, 32.91, 11.90, 2.77, 0.70]]
#optionalprice matrix for maturity x K
P = np.asarray(P)
[K, tau] = np.meshgrid(K, tau)

sigma = calcbsimpvol(dict(cp=cp, P=P, S=S, K=K, tau=tau, r=r, q=q))
print(sigma)    
 """   