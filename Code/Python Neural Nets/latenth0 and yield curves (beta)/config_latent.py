# config file

### Preambel
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

### Data Import
mat         = scipy.io.loadmat("id_2c1fb86638da4e61_data_vola_27944.mat")
data        = mat['data_vola']
"""assume data is of structure N*(Nparameters+1(h0)+Nmaturitites(yieldcurve)+Nstrikes*Nmaturites(grid))""" 
Nparameters = 4
maturities  = np.array([30, 60, 90, 120, 150, 180, 210])
strikes     = np.array([0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1])
Nstrikes    = len(strikes)   
Nmaturities = len(maturities)   
xx          = data[:,:Nparameters]
yy          = data[:,Nparameters:]
ub=np.amax(xx, axis=0)
lb=np.amin(xx, axis=0)
diff = ub-lb
bound_sum =ub+lb
### Trainset generation
X_train, X_test, y_train, y_test = train_test_split(
    xx, yy, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.15, random_state=42)

Ntest= X_test.shape[0]
Ntrain= X_train.shape[0]
Nval= X_val.shape[0]

def ytransform(y_train,y_val,y_test):
    #return [scale.transform(y_train),scale.transform(y_val), 
    #        scale.transform(y_test)]
    return [y_train,y_val,y_test]

def yinversetransform(y):
    return y
    #return scale.inverse_transform(y)
    
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



# Splitting the sample correctly
[y_train_trafo, y_val_trafo, y_test_trafo]=ytransform(y_train, y_val, y_test)
h0_train_trafo    = y_train_trafo[:,0]
h0_val_trafo      = y_val_trafo[:,0]
h0_test_trafo     = y_test_trafo[:,0]
rates_train_trafo = y_train_trafo[:,[1,2,3,4,5,6,7]]
rates_val_trafo   = y_val_trafo[:,[1,2,3,4,5,6,7]]
rates_test_trafo  = y_test_trafo[:,[1,2,3,4,5,6,7]]
y_train_trafo = y_train_trafo[:,8:]
y_val_trafo   = y_val_trafo[:,8:]
y_test_trafo  = y_test_trafo[:,8:]

# reshaping for NN1: Pricer
y_train_trafo1 = np.asarray([y_train_trafo[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
y_val_trafo1   =  np.asarray([y_val_trafo[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])
y_test_trafo1  =  np.asarray([y_test_trafo[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])
X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo   = np.array([myscale(x) for x in X_val])
X_test_trafo  = np.array([myscale(x) for x in X_test])
X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo   = np.array([myscale(x) for x in X_val])
X_test_trafo  = X_test_trafo.reshape((Ntest,Nparameters,1,1))
X_train_trafo = X_train_trafo.reshape((Ntrain,Nparameters,1,1))
X_val_trafo   = X_val_trafo.reshape((Nval,Nparameters,1,1))

# reshaping for NN2: Calibration
y_train_trafo2 = y_train_trafo.reshape((Ntrain,Nmaturities,Nstrikes,1))
y_test_trafo2  = y_test_trafo.reshape((Ntest,Nmaturities,Nstrikes,1))
y_val_trafo2   = y_val_trafo.reshape((Nval,Nmaturities,Nstrikes,1))
X_val_trafo2   = X_val_trafo.reshape((Nval,Nparameters))
X_train_trafo2 = X_train_trafo.reshape((Ntrain,Nparameters))
X_test_trafo2  = X_test_trafo.reshape((Ntest,Nparameters))
