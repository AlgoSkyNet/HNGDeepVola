# config file
# This file preprocesses the underlying price and volatility data for later usage
# in the neural networks
### Preambel
import scipy.io
import numpy as np
import numpy.matlib as npm
from sklearn.model_selection import train_test_split
#import sys
#import os
#from sklearn.preprocessing import StandardScaler


### Data Import
path = "C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Python Neural Nets/9x9 Grid_Moneyness/Dataset/"
#path = "D:\GitHub\MasterThesisHNGDeepVola\Code\Python Neural Nets\9x9 Grid_Moneyness/Dataset/"

#name_price = "id_Moneyness_8cd524ab1cd9408b_data_price_norm_53392_bigprice.mat"#"id_aa11a111a1aa1a1a_data_price_norm_143032.mat"
#name_vola  = "id_Moneyness_8cd524ab1cd9408b_data_vola_norm_53392_bigprice.mat"
#name_vega  = "id_Moneyness_8cd524ab1cd9408b_data_vega_norm_53392_bigprice.mat"
#name_price = "Moneyness_price_norm_81509_1e-2.mat"
#name_vola  = "Moneyness_vola_norm_81509_1e-2.mat"
#name_vega  = "Moneyness_vega_norm_81509_1e-2.mat"
name_price = "Moneyness_price_norm_400_1e-6.mat"
name_vola  = "Moneyness_vola_norm_400_1e-6.mat"
name_vega  = "Moneyness_vega_norm_400_1e-6.mat"

mat         = scipy.io.loadmat(path+name_vola)
data_vola   = mat['data_vola']
mat         = scipy.io.loadmat(path+name_price)
data_price  = mat['data_price']
mat         = scipy.io.loadmat(path+name_vega)
vega        = mat['data_vega']


### Initialisation
S0              = 2000
Nparameters     = 5
maturities      = np.array([10,40,70,100,130,160,190,220,250])
moneyness       = np.array([1.1, 1.075, 1.05, 1.025, 1, 0.975, 0.95, 0.925, 0.9])
strikes         = 1/moneyness
Nstrikes        = len(strikes)   
Nmaturities     = len(maturities) 
Ntotal          = data_price.shape[0]
rates           = data_price[:,Nparameters:Nparameters+Nmaturities]
# parameters
parameters      = data_price[:,:Nparameters]
ub              = np.amax(parameters,axis=0)
lb              = np.amin(parameters,axis=0)
# vola  
vola            = data_vola[:,Nparameters+Nmaturities:]
ub_vola         = np.amax(vola)
lb_vola         = np.amin(vola)
diff_vola       = ub_vola-lb_vola
bound_sum_vola  = ub_vola+lb_vola
# price
price           = data_price[:,Nparameters+Nmaturities:]
ub_price        = np.amax(price)
lb_price        = np.amin(price)
diff_price      = ub_price-lb_price
bound_sum_price = ub_price+lb_price

def ytransform(y,specs=99):
    if specs == 99:
        #no scale works best for vola
        return y
    elif specs == 1:
        #vola minmax -1,1
        return (y - (ub_vola + lb_vola)*0.5) * 2 / (ub_vola - lb_vola)
    elif specs == 0: 
        #price minmax -1,1
        return (y - (ub_price + lb_price)*0.5) * 2 / (ub_price - lb_price)
    elif specs == 2: 
        #price minmax shift 0,1
        return 0.5*((y - (ub_price + lb_price)*0.5) * 2 / (ub_price - lb_price) +1)  

def yinversetransform(y,specs = 99):
    if specs == 99:
        #no scale works best for vola
        return y
    elif specs == 1:
        #vola minmax -1,1
        return y*(ub_vola - lb_vola) *0.5 + (ub_vola + lb_vola)*0.5
    elif specs == 0:
        #price minmax -1,1
        return y*(ub_price - lb_price) *0.5 + (ub_price + lb_price)*0.5
    elif specs == 2:
        #price minmax shift 0,1
        return (2*y-1)*(ub_price - lb_price) *0.5 + (ub_price + lb_price)*0.5
    
def myscale(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=(x[i] - (ub[i] + lb[i])*0.5) * 2 / (ub[i] - lb[i])
    return res

def myinverse(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=x[i]*(ub[i] - lb[i]) *0.5 + (ub[i] + lb[i])*0.5
    return res

### Split indecee
idx_train,idx_test = train_test_split(list(range(Ntotal)), test_size=0.15, random_state=42)
idx_train,idx_val  = train_test_split(idx_train, test_size=0.15, random_state=42)
Ntest              = len(idx_test)
Ntrain             = len(idx_train)
Nval               = len(idx_val)

### Trafo 
vola_trafo         = ytransform(vola,1)
price_trafo         = ytransform(price,2)
parameters_trafo   = np.array([myscale(x) for x in parameters])

strike_net     = S0*npm.repmat(np.asarray(strikes).reshape(1,9), 9,1)
maturity_net   = 1/252*npm.repmat(np.asarray(maturities).reshape(9,1), 1,9)
intrinsic_net  = []
rates_net      = []

print("Generating Data....") 
for i in range(Ntotal):
    if i%10000 == 0:
        print(round(i/Ntotal*100,1))
    rates_tmp  = npm.repmat(rates[i,:].reshape((9,1)),1,9)
    tmp        = S0-np.exp(-rates_tmp*maturity_net)*strike_net
    tmp[tmp<0] = 0
    rates_net.append(rates_tmp)
    intrinsic_net.append(tmp)

### Reshape
price_1            = price.reshape((Ntotal,Nmaturities,Nstrikes))
price_trafo_1      = price_trafo.reshape((Ntotal,Nmaturities,Nstrikes))
vola_1             = vola.reshape((Ntotal,Nmaturities,Nstrikes))
vola_trafo_1       = vola_trafo.reshape((Ntotal,Nmaturities,Nstrikes))
vega_1             = vega.reshape((Ntotal,Nmaturities,Nstrikes))
intrinsic_net      = np.asarray(intrinsic_net).reshape(Ntotal,9,9)
rates_net          = np.asarray(rates_net).reshape(Ntotal,9,9)

print("Data Import complete.") 