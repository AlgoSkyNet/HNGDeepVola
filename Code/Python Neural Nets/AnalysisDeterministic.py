# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:54:34 2020

@author: Henrik
"""

# In[Initialisation]
# This Initialisation will be used for everyfile to ensure the same conditions everytime!
# Preambel
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.compat.v1.keras.layers import InputLayer,Dense,Flatten, Conv2D, Dropout, Input,ZeroPadding2D,MaxPooling2D
from tensorflow.compat.v1.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize,NonlinearConstraint
#import matplotlib.lines as mlines
#import matplotlib.transforms as mtransforms
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import scipy
import scipy.io
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import random
#import time
#import keras

# import data set
from config import data,Nparameters,maturities,strikes,Nstrikes,Nmaturities,Ntest,Ntrain,Nval
from config import xx,yy,ub,lb,diff,bound_sum, X_train,X_test,X_val,y_train,y_test,y_val
from config import y_train_trafo,y_val_trafo,y_test_trafo,X_train_trafo,X_val_trafo,X_test_trafo
from config import y_train_trafo2,y_val_trafo2,y_test_trafo2,X_train_trafo2,X_val_trafo2,X_test_trafo2
# import custom functions #scaling tools
from add_func import ytransform, yinversetransform,myscale, myinverse

#custom errors
from add_func import root_mean_squared_error,root_relative_mean_squared_error,mse_constraint,rmse_constraint
#else
from add_func import constraint_violation,pricing_plotter

tf.compat.v1.keras.backend.set_floatx('float64')  

from add_func import ownTimer
t = ownTimer()



S0=1.
y_test_re    = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))


data_deterministic = scipy.io.loadmat('optiparams.mat')
X_test_deter = data_deterministic["opti_params"][0:Ntest,:]

prediction_invtrafo= X_test_deter
prediction_std = np.std(X_test_deter,axis=0)
error = np.zeros((Ntest,Nparameters))
for i in range(Ntest):
    error[i,:] =  np.abs((X_test[i,:]-X_test_deter[i,:])/X_test[i,:])
err1 = np.mean(error,axis = 0)
err2 = np.median(error,axis = 0)
err_std = np.std(error,axis = 0)
idx = np.argsort(error[:,0], axis=None)
good_idx = idx[:-100]
    
_,_,c =constraint_violation(X_test_deter)
_,_,c2 =constraint_violation(X_test)


testing_violation = c>=1
testing_violation2 = (c<1)
vio_error = error[testing_violation,:]
vio_error2 = error[testing_violation2,:]


plt.figure(figsize=(14,4))
ax=plt.subplot(1,3,1)
plt.boxplot(error)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors")
ax=plt.subplot(1,3,2)
plt.boxplot(vio_error)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors parameter violation")
ax=plt.subplot(1,3,3)
plt.boxplot(vio_error2)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors no parameter violation")
plt.show()

print("violation error mean in %:",100*np.mean(vio_error,axis=0))
print("no violation error mean in %:",100*np.mean(vio_error2,axis=0))
print("violation error median in %:",100*np.median(vio_error,axis=0))
print("no violation error median in %:",100*np.median(vio_error2,axis=0))
print("error mean in %:",100*err1)
print("error median in %:",100*err2)

fig = plt.figure()
plt.scatter(c2,c)
plt.plot(np.arange(0, np.max(c),0.5),np.arange(0, np.max(c),0.5),'-r')
plt.xlabel("True Constraint")
plt.ylabel("Forecasted Constraint")


plt.figure(figsize=(14,4))
ax=plt.subplot(1,5,1)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
plt.scatter(c2[testing_violation],vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,2)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
plt.scatter(c2[testing_violation],vio_error[:,1],c="b",s=1,marker="x",label="beta con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,3)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
plt.scatter(c2[testing_violation],vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,4)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
plt.scatter(c2[testing_violation],vio_error[:,3],c="b",s=1,marker="x",label="omega con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,5)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,4],c="r",s=1,marker="x",label="sigma0 no con")
plt.scatter(c2[testing_violation],vio_error[:,4],c="b",s=1,marker="x",label="sigma0 con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
fig = plt.figure()
plt.scatter(c2,c)
plt.plot(np.arange(0, np.max(c),0.5),np.arange(0, np.max(c),0.5),'-r')
plt.xlabel("True Constraint")
plt.ylabel("Forecasted Constraint")

plt.figure(figsize=(14,4))
ax=plt.subplot(1,5,1)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,2)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,1],c="b",s=1,marker="x",label="beta con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,3)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,4)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,3],c="b",s=1,marker="x",label="omega con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,5)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,4],c="r",s=1,marker="x",label="sigma0 no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,4],c="b",s=1,marker="x",label="sigma0 con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
