# config file

### Preambel
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler



### Data Import
mat         = scipy.io.loadmat("id_3283354135d44b67_data_vola_norm_231046clean.mat")
data_vola        = mat['data_vola']
mat         = scipy.io.loadmat("id_3283354135d44b67_data_price_norm_231046clean.mat")
data_price       = mat['data_price']
""" data is of structure N*(Nparameters+1(h0)+Nmaturitites(yieldcurve)+Nstrikes*Nmaturites(grid))""" 

### Initialisation
Nparameters     = 5
maturities      = np.array([30, 60, 90, 120, 150, 180, 210])
strikes         = np.array([0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1])
Nstrikes        = len(strikes)   
Nmaturities     = len(maturities) 
xx              = data_vola[:,:Nparameters]
ub              = np.amax(xx, axis=0)
lb              = np.amin(xx, axis=0)
diff            = ub-lb
bound_sum       = ub+lb
# vola  
yy              = data_vola[:,Nparameters:]
ub_vola         = np.amax(yy)
lb_vola         = np.amin(yy)
diff_vola       = ub_vola-lb_vola
bound_sum_vola  = ub_vola+lb_vola
# price
yy_price        = data_price[:,Nparameters:]
ub_price        = np.amax(yy_price)
lb_price        = np.amin(yy_price)
diff_price      = ub_price-lb_price
bound_sum_price = ub_price+lb_price
#concatenate
y_all           = np.concatenate((yy,yy_price),axis =1)

def ytransform(y,specs=99):
    if specs == 99:
        #no scale works best for vola
        return y
    elif specs == 1:
        #vola minmax
        return (y - (ub_vola + lb_vola)*0.5) * 2 / (ub_vola - lb_vola)
    elif specs ==0: 
        #price
        return (y - (ub_price + lb_price)*0.5) * 2 / (ub_price - lb_price)
    
def yinversetransform(y,specs = 99):
    if specs == 99:
        #no scale works best for vola
        return y
    elif specs == 1:
        #vola minmax
        return y*(ub_vola - lb_vola) *0.5 + (ub_vola + lb_vola)*0.5
    elif specs == 0:
        #price minmax
        return y*(ub_price - lb_price) *0.5 + (ub_price + lb_price)*0.5

    
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

### Trainset generation
X_train, X_test, y_train, y_test = train_test_split(
    xx, y_all, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.15, random_state=42)
rates_train   = y_train[:,:Nmaturities]
rates_val     = y_val[:,:Nmaturities]
rates_test    = y_test[:,:Nmaturities]
y_train_price = y_train[:,Nmaturities*(Nstrikes+2):]
y_train       = y_train[:,Nmaturities:Nmaturities*(Nstrikes+1)]
y_test_price  = y_test[:,Nmaturities*(Nstrikes+2):]
y_test        = y_test[:,Nmaturities:Nmaturities*(Nstrikes+1)]
y_val_price   = y_val[:,Nmaturities*(Nstrikes+2):]
y_val         = y_val[:,Nmaturities:Nmaturities*(Nstrikes+1)]

Ntest= X_test.shape[0]
Ntrain= X_train.shape[0]
Nval= X_val.shape[0]

## Splitting the sample correctly

# vola
y_train_trafo = ytransform(y_train)
y_val_trafo   = ytransform(y_val)
y_test_trafo  = ytransform(y_test)

#price
y_train_trafo_price = ytransform(y_train_price,0)
y_val_trafo_price   = ytransform(y_val_price,0)
y_test_trafo_price  = ytransform(y_test_price,0)


## reshaping for NN1: Pricer
X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo   = np.array([myscale(x) for x in X_val])
X_test_trafo  = np.array([myscale(x) for x in X_test])
X_test_trafo  = X_test_trafo.reshape((Ntest,Nparameters,1,1))
X_train_trafo = X_train_trafo.reshape((Ntrain,Nparameters,1,1))
X_val_trafo   = X_val_trafo.reshape((Nval,Nparameters,1,1))
# vola
y_train_trafo1 = np.asarray([y_train_trafo[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
y_val_trafo1   =  np.asarray([y_val_trafo[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])
y_test_trafo1  =  np.asarray([y_test_trafo[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])
# price
y_train_trafo1_price = np.asarray([y_train_trafo_price[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
y_val_trafo1_price   =  np.asarray([y_val_trafo_price[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])
y_test_trafo1_price =  np.asarray([y_test_trafo_price[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])


## reshaping for NN2: Calibration
X_val_trafo2   = X_val_trafo.reshape((Nval,Nparameters))
X_train_trafo2 = X_train_trafo.reshape((Ntrain,Nparameters))
X_test_trafo2  = X_test_trafo.reshape((Ntest,Nparameters))
# vola
y_train_trafo2 = y_train_trafo.reshape((Ntrain,Nmaturities,Nstrikes,1))
y_test_trafo2  = y_test_trafo.reshape((Ntest,Nmaturities,Nstrikes,1))
y_val_trafo2   = y_val_trafo.reshape((Nval,Nmaturities,Nstrikes,1))
# price
y_train_trafo2_price = y_train_trafo_price.reshape((Ntrain,Nmaturities,Nstrikes,1))
y_test_trafo2_price  = y_test_trafo_price.reshape((Ntest,Nmaturities,Nstrikes,1))
y_val_trafo2_price   = y_val_trafo_price.reshape((Nval,Nmaturities,Nstrikes,1))
