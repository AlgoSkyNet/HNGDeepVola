#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# This code gives a short summary on the current progress (last update: 20.01.2020).
# In the following a CNN aswell as a FFNN are trained to learn the mapping HNG-Parameters (a,b,g*,w,h0) to HNG-Volatility surface. A first approach on  training the inverse mapping with CNN is given. 
# For the used dataset (50.000 szenarios), each szenario is generated as followed:
# S&P500 returns are used to get optimal HNG parameters for each week in 2015 (10years rolling window). The initial variance is used as variable aswell.
# Afterwards for each week in 2015 Call Prices are used to get optimal HNG parameters. The initial variance is set to the corrensponding value from MLE optimization.
#  Those 52 optimal parameters combinations are then used as bounds. 
#  To generate a szenario a random value between the bounds is uniformly choosen (initial variance included)
#  and a grid of implied variance is calculated for Maturities [30, 60, 90, 120, 150, 180, 210] days and Moneyness     [0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1].
# 
# h_0 is set fixed to the MLE parameter


# In[1. Initialisation/ Preambel and Data Import]:
# This Initialisation will be used for everyfile to ensure the same conditions everytime!
#def names_data():
#    return name_price,name_vola




import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.compat.v1.keras.layers import Reshape,InputLayer,Dense,Flatten, Conv2D,Conv1D, Dropout, Input,ZeroPadding2D,ZeroPadding1D,MaxPooling2D
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#from scipy.optimize import minimize,NonlinearConstraint
#import matplotlib.lines as mlines
#import matplotlib.transforms as mtransforms
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mpl_toolkits.mplot3d import Axes3D  
#from matplotlib import cm
import scipy
#import scipy.io
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
#import random
#import time
#import keras
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#import keras
#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 0 , 'CPU': 20} ) 
#sess = tf.compat.v1.Session(config=config) 
#tf.compat.v1.keras.backend.set_session(sess)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
## import data set
from config_9x9 import Nparameters,maturities,strikes,Nstrikes,Nmaturities,Ntest,Ntrain,Nval
from config_9x9 import xx,rates_train,rates_val,rates_test,ub,lb,diff,bound_sum
from config_9x9 import X_train,X_test,X_val,X_train_trafo,X_val_trafo,X_test_trafo,X_train_trafo2,X_val_trafo2,X_test_trafo2
# vola
from config_9x9 import yy,y_train,y_test,y_val,ub_vola,lb_vola,diff_vola,bound_sum_vola
from config_9x9 import y_train_trafo,y_val_trafo,y_test_trafo
from config_9x9 import y_train_trafo1,y_val_trafo1,y_test_trafo1
from config_9x9 import y_train_trafo2,y_val_trafo2,y_test_trafo2
# price
from config_9x9 import yy_price,y_train_price,y_test_price,y_val_price,ub_price,lb_price,diff_price,bound_sum_price
from config_9x9 import y_train_trafo_price,y_val_trafo_price,y_test_trafo_price
from config_9x9 import y_train_trafo1_price,y_val_trafo1_price,y_test_trafo1_price
from config_9x9 import y_train_trafo2_price,y_val_trafo2_price,y_test_trafo2_price
from config_9x9 import vega_train,vega_test,vega_val

# import custom functions #scaling tools
from config_9x9 import ytransform, yinversetransform,myscale,myinverse

#custom errors
from add_func_9x9 import root_mean_squared_error,root_relative_mean_squared_error,mse_constraint,rmse_constraint
#else
from add_func_9x9 import constraint_violation,pricing_plotter,plotter_autoencoder

tf.compat.v1.keras.backend.set_floatx('float64')  



def autoencoder(nn1,nn2):
    def autoencoder_predict(y_values):
        prediction = nn2.predict(y_values)
        prediction_trafo = prediction.reshape((Ntest,Nparameters,1,1))
        forecast = nn1.predict(prediction_trafo).reshape(Ntest,Nmaturities,Nstrikes)
        return forecast
    return autoencoder_predict



# In[2.3 CNN as Encoder / OPTIOP LIKELHOODPricing Kernel without riskfree rate]:

def option_likelyhood(y_true_with_vega, y_pred):
        return K.mean(K.square((y_pred - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:]))  

NN1price = Sequential() 
NN1price.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1price.add(ZeroPadding2D(padding=(2, 2)))
NN1price.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1price.add(ZeroPadding2D(padding=(3,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,2)))
NN1price.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,1)))
NN1price.add(Conv2D(32, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1price.add(MaxPooling2D(pool_size=(2, 1)))
#NN1price.add(Dropout(0.25))
#NN1price.add(ZeroPadding2D(padding=(0,1)))
NN1price.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1price.add(MaxPooling2D(pool_size=(4, 1)))
NN1price.summary()
#NN1price.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN1price.compile(loss =option_likelyhood, optimizer = "adam",metrics=["MAPE","MSE"])

vega_train1 = np.asarray([vega_train[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
vega_test1 = np.asarray([vega_test[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])
vega_val1 = np.asarray([vega_val[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])

y_train_tmp = np.concatenate((y_train_trafo1_price,vega_train1),axis=1)
y_test_tmp = np.concatenate((y_test_trafo1_price,vega_test1),axis=1)
y_val_tmp = np.concatenate((y_val_trafo1_price,vega_val1),axis=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)

NN1price.fit(X_train_trafo, y_train_tmp, batch_size=64, validation_data = (X_val_trafo, y_val_tmp), epochs = 1000, verbose = True, shuffle=1,callbacks=[es])
NN1price.save_weights("pricerweights_norate_optll.h5")#id_3283354135d44b67_data_price_norm_231046clean
#NN1price.load_weights("pricerweights_norate_optll.h5")#id_3283354135d44b67_data_price_norm_231046clean

#  Results 
# The following plots show the performance on the testing set
S0=1.
#y_test_re    = yinversetransform(y_test_tmp,0).reshape((Ntest-1500,Nmaturities,Nstrikes))
#prediction   = NN1c.predict(X_test_tmp).reshape((Ntest-1500,Nmaturities,Nstrikes))

y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = NN1price.predict(X_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
err_matrix = np.mean(err_rel_mat,axis=(1,2))
err_idx = np.argsort(err_matrix)
plt.figure(figsize= (14,4))
#plt.plot(np.min(y_test_re,axis=(1,2)))
#plt.plot(err_matrix)
plt.yscale("log")
plt.xscale("log")
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()




plt.figure(figsize= (14,4))
#plt.plot(np.min(y_test_re,axis=(1,2)))
#plt.plot(err_matrix)
plt.xscale("log")
plt.yscale("log")
plt.hist(err_rel_mat.flatten(),bins=100)
plt.show()



# In[2.3 CNN as Encoder / Pricing Kernel with riskfree rate]:
NN1price2 = Sequential() 
NN1price2.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NN1price2.add(ZeroPadding2D(padding=(2, 2)))
NN1price2.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2.add(ZeroPadding2D(padding=(2,2)))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
#NN1price2.add(Dropout(0.25))
NN1price2.add(ZeroPadding2D(padding=(2,2)))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2.add(ZeroPadding2D(padding=(2,2)))
NN1price2.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price2.add(ZeroPadding2D(padding=(2,2)))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2.add(ZeroPadding2D(padding=(2,2)))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NN1price2.add(ZeroPadding2D(padding=(2,2)))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(1,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1price2.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
NN1price2.add(Conv2D(4, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1price2.summary()

#setting
#NN1price2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN1price2.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE"])
inputs_train =np.concatenate((X_train_trafo,rates_train.reshape((Ntrain,Nmaturities,1,1))),axis=1)
inputs_val = np.concatenate((X_val_trafo,rates_val.reshape((Nval,Nmaturities,1,1))),axis=1)
inputs_test = np.concatenate((X_test_trafo,rates_test.reshape((Ntest,Nmaturities,1,1))),axis=1)
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)

#NN1price2.fit(inputs_train, y_train_trafo1_price[:,:,:,[5,6,7,8]], batch_size=64, validation_data = (inputs_val, y_val_trafo1_price[:,:,:,[5,6,7,8]]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NN1price2.save_weights("price_weights_rate_9x9.h5")
NN1price2.load_weights("price_weights_rate_9x9.h5")

#  Results 
# The following plots show the performance on the testing set
S0=1.
y_test_re    = yinversetransform(y_test_trafo1_price[:,:,:,[5,6,7,8]]).reshape((Ntest,Nmaturities,4))
prediction   = NN1price2.predict(inputs_test).reshape((Ntest,Nmaturities,4))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
err_matrix = np.mean(err_rel_mat,axis=(1,2))
err_idx = np.argsort(err_matrix)
#plt.plot(err_matrix)
plt.figure(figsize=(14,4))
mini = np.min(y_test_price,axis=1)
mini_idx = mini>10e-5
tmp_test = np.argsort(mini[mini_idx])
plt.plot(err_matrix[tmp_test])
plt.show()
err_median = np.median(err_rel_mat,axis=0)
err_mean = np.median(err_rel_mat,axis=0)


plt.figure(figsize= (14,4))
#plt.xscale("log")
plt.yscale("log")
plt.hist(err_rel_mat.flatten(),bins=100)
plt.show()

plt.figure(figsize= (14,4))
#plt.yscale("log")
#plt.xscale("log")
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()



NN1price2a = Sequential() 
NN1price2a.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NN1price2a.add(ZeroPadding2D(padding=(2, 2)))
NN1price2a.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2a.add(ZeroPadding2D(padding=(2,2)))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
#NN1price2a.add(Dropout(0.25))
NN1price2a.add(ZeroPadding2D(padding=(2,2)))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2a.add(ZeroPadding2D(padding=(2,2)))
NN1price2a.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price2a.add(ZeroPadding2D(padding=(2,2)))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2a.add(ZeroPadding2D(padding=(2,2)))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NN1price2a.add(ZeroPadding2D(padding=(2,2)))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2a.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(1,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1price2a.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
NN1price2a.add(Conv2D(5, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1price2a.summary()

#setting
#NN1price2a.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN1price2a.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE"])
inputs_train =np.concatenate((X_train_trafo,rates_train.reshape((Ntrain,Nmaturities,1,1))),axis=1)
inputs_val = np.concatenate((X_val_trafo,rates_val.reshape((Nval,Nmaturities,1,1))),axis=1)
inputs_test = np.concatenate((X_test_trafo,rates_test.reshape((Ntest,Nmaturities,1,1))),axis=1)
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)

#NN1price2a.fit(inputs_train, y_train_trafo1_price[:,:,:,[0,1,2,3,4]], batch_size=64, validation_data = (inputs_val, y_val_trafo1_price[:,:,:,[0,1,2,3,4]]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NN1price2a.save_weights("priceweights_rates_left2a.h5")#id_3283354135d44b67_data_price_norm_231046clean
NN1price2a.load_weights("priceweights_rates_left2a.h5")#id_3283354135d44b67_data_price_norm_231046clean

#  Results 
# The following plots show the performance on the testing set
S0=1.
#y_test_re    = yinversetransform(y_test_tmp,0).reshape((Ntest-1500,Nmaturities,Nstrikes))
#prediction   = NN1c.predict(X_test_tmp).reshape((Ntest-1500,Nmaturities,Nstrikes))

y_test_re    = yinversetransform(y_test_trafo1_price[:,:,:,[0,1,2,3,4]]).reshape((Ntest,Nmaturities,5))
prediction_a   = NN1price2a.predict(inputs_test).reshape((Ntest,Nmaturities,5))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction_a,y_test_re)

err_idx = np.argsort(err_matrix)
plt.figure(figsize= (14,4))
#plt.plot(np.min(y_test_re,axis=(1,2)))
#plt.plot(err_matrix)
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()


# IDEE link oben 2a und rest 2






NN1price2b = Sequential() 
NN1price2b.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NN1price2b.add(ZeroPadding2D(padding=(2, 2)))
NN1price2b.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2b.add(ZeroPadding2D(padding=(2,2)))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
#NN1price2b.add(Dropout(0.25))
NN1price2b.add(ZeroPadding2D(padding=(2,2)))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2b.add(ZeroPadding2D(padding=(2,2)))
NN1price2b.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price2b.add(ZeroPadding2D(padding=(2,2)))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2b.add(ZeroPadding2D(padding=(2,2)))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NN1price2b.add(ZeroPadding2D(padding=(2,2)))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price2b.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(1,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1price2b.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
NN1price2b.add(Conv2D(2, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1price2b.summary()

#setting
NN1price2b.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NN1price2b.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE"])
inputs_train =np.concatenate((X_train_trafo,rates_train.reshape((Ntrain,Nmaturities,1,1))),axis=1)
inputs_val = np.concatenate((X_val_trafo,rates_val.reshape((Nval,Nmaturities,1,1))),axis=1)
inputs_test = np.concatenate((X_test_trafo,rates_test.reshape((Ntest,Nmaturities,1,1))),axis=1)
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)

#NN1price2b.fit(inputs_train, y_train_trafo1_price[:,:,:,[7,8]], batch_size=64, validation_data = (inputs_val, y_val_trafo1_price[:,:,:,[7,8]]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NN1price2b.save_weights("priceweights_rates_left2b.h5")#id_3283354135d44b67_data_price_norm_231046clean
NN1price2b.load_weights("priceweights_rates_left2b.h5")#id_3283354135d44b67_data_price_norm_231046clean
S0=1.
#y_test_re    = yinversetransform(y_test_tmp,0).reshape((Ntest-1500,Nmaturities,Nstrikes))
#prediction   = NN1c.predict(X_test_tmp).reshape((Ntest-1500,Nmaturities,Nstrikes))

y_test_re    = yinversetransform(y_test_trafo1_price[:,:,:,[7,8]]).reshape((Ntest,Nmaturities,2))
prediction_b   = NN1price2b.predict(inputs_test).reshape((Ntest,Nmaturities,2))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction_b,y_test_re)

err_idx = np.argsort(err_matrix)
plt.figure(figsize= (14,4))
#plt.plot(np.min(y_test_re,axis=(1,2)))
#plt.plot(err_matrix)
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()
err_mean = np.mean(err_rel_mat,axis=0)



#prediction_full = np.zeros_like(prediction)
#prediction_full[:,:,np.array([0,1,2,3,4])] = prediction_a
#prediction_full[:,-1,:] = prediction[:,-1,:]
#prediction_full[:,:,np.array([5,6,7,8])] = prediction[:,:,np.array([5,6,7,8])]
prediction_full = np.concatenate((prediction_a,prediction),axis = 2)
prediction_full[:,:,[7,8]] = prediction_b

y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))


err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction_full,y_test_re)
err_matrix = np.mean(err_rel_mat,axis=(1,2))
err_idx = np.argsort(err_matrix)
#plt.plot(err_matrix)
plt.figure(figsize=(14,4))
mini = np.min(y_test_price,axis=1)
mini_idx = mini>10e-5
tmp_test = np.argsort(mini[mini_idx])
plt.plot(err_matrix[tmp_test])
plt.show()
err_median = np.median(err_rel_mat,axis=0)
err_mean = np.mean(err_rel_mat,axis=0)
err_max = np.max(err_rel_mat,axis =0)

plt.figure(figsize= (14,4))
#plt.xscale("log")
plt.yscale("log")
plt.hist(err_rel_mat.flatten(),bins=100)
plt.show()

plt.figure(figsize= (14,4))
#plt.yscale("log")
#plt.xscale("log")
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()

from matplotlib.colors import LogNorm
plt.figure(figsize= (14,4))
ax = plt.subplot(1,1,1)
plt.imshow(err_mean,norm=LogNorm(vmin=err_mean.min(), vmax=err_mean.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)

plt.show()


"""
prediction   = NN1price2.predict(inputs_test).reshape((Ntest,Nmaturities,4))
prediction_a   = NN1price2a.predict(inputs_test).reshape((Ntest,Nmaturities,5))
prediction_b   = NN1price2b.predict(inputs_test).reshape((Ntest,Nmaturities,2))
prediction_full = np.concatenate((prediction_a,prediction),axis = 2)
prediction_full[:,:,[7,8]] = prediction_b


new_val = NN1price2.predict(inputs_val).reshape((Nval,Nmaturities,4))
new_val_a = NN1price2a.predict(inputs_val).reshape((Nval,Nmaturities,5))
new_val_b = NN1price2b.predict(inputs_val).reshape((Nval,Nmaturities,2))
new_val_full =np.concatenate((new_val_a,new_val),axis = 2)
new_val_full[:,:,[7,8]] = new_val_b


new_train = NN1price2.predict(inputs_train).reshape((Ntrain,Nmaturities,4))
new_train_a = NN1price2a.predict(inputs_train).reshape((Ntrain,Nmaturities,5))
new_train_b = NN1price2b.predict(inputs_train).reshape((Ntrain,Nmaturities,2))
new_train_full =np.concatenate((new_train_a,new_train),axis = 2)
new_train_full[:,:,[7,8]] = new_train_b




NN1price2full = Sequential() 
NN1price2full.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1,)))
NN1price2full.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(2,2),activation='sigmoid'))#X_train_trafo.shape[1:],activation='elu'))
NN1price2full.add(ZeroPadding2D(padding=(1,1)))
#NN1price2full.add(Conv2D(64, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='sigmoid'))
NN1price2full.add(ZeroPadding2D(padding=(1,1)))
NN1price2full.add(Conv2D(96, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='sigmoid'))
#NN1price2full.add(ZeroPadding2D(padding=(2,2)))
NN1price2full.add(Conv2D(96, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='sigmoid'))
NN1price2full.add(Reshape((24,24,6)))
NN1price2full.add(Conv2D(32, (3, 3),padding='valid',use_bias =True,strides =(2,2),activation ='sigmoid'))
NN1price2full.add(Conv2D(1, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='sigmoid'))
#NN1price2full.summary()
NN1price2full.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#NN1price2full.fit(new_train_full.reshape((Ntrain,Nmaturities,Nstrikes,1)), y_train_trafo2_price, batch_size=64, validation_data = (new_val_full.reshape((Nval,Nmaturities,Nstrikes,1)), y_val_trafo2_price), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NN1price2b.save_weights("priceweights_rates_left2a.h5")#id_3283354135d44b67_data_price_norm_231046clean
#NN1price2b.load_weights("priceweights_rates_left2a.h5")#id_3283354135d44b67_data_price_norm_231046clean
S0=1.



y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))
prediction_new = NN1price2full.predict(prediction_full.reshape((Ntest,Nmaturities,Nstrikes,1))).reshape((Ntest,Nmaturities,Nstrikes))

err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction_new,y_test_re)
err_matrix = np.mean(err_rel_mat,axis=(1,2))
err_idx = np.argsort(err_matrix)
#plt.plot(err_matrix)
plt.figure(figsize=(14,4))
mini = np.min(y_test_price,axis=1)
mini_idx = mini>10e-5
tmp_test = np.argsort(mini[mini_idx])
plt.plot(err_matrix[tmp_test])
plt.show()
err_median = np.median(err_rel_mat,axis=0)
err_mean = np.mean(err_rel_mat,axis=0)


plt.figure(figsize= (14,4))
#plt.xscale("log")
plt.yscale("log")
plt.hist(err_rel_mat.flatten(),bins=100)
plt.show()

plt.figure(figsize= (14,4))
#plt.yscale("log")
#plt.xscale("log")
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()


"""





# In[2.4 CNN as Encoder / Pricing Kernel with no riskfree rate]:
"""
NN1price = Sequential() 
NN1price.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1price.add(ZeroPadding2D(padding=(2, 2)))
NN1price.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1price.add(ZeroPadding2D(padding=(3,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,2)))
NN1price.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1price.add(MaxPooling2D(pool_size=(2, 1)))
#NN1price.add(Dropout(0.25))
#NN1price.add(ZeroPadding2D(padding=(0,1)))
NN1price.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1price.add(MaxPooling2D(pool_size=(4, 1)))
NN1price.summary()

NN1price.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])


#NN1price.compile(loss ="MSE", optimizer = "adam",metrics=["MAPE","MSE"])
NN1price.fit(X_train_trafo, y_train_trafo1_price, batch_size=64, validation_data = (X_val_trafo, y_val_trafo1_price), epochs = 1000, verbose = True, shuffle=1,callbacks=[es])
#NN1c.save_weights("pricerweights_noriskfreerate_price_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
#NN1c.load_weights("pricerweights_noriskfreerate_price_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean


#  Results 
# The following plots show the performance on the testing set
S0=1.
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = NN1price.predict(X_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
err_matrix = np.mean(err_rel_mat,axis=(1,2))
err_idx = np.argsort(err_matrix)

plt.figure(figsize= (14,4))
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()

plt.figure(figsize= (14,4))
#plt.xscale("log")
plt.yscale("log")
plt.hist(err_rel_mat.flatten(),bins=100)
plt.show()

plt.figure(figsize= (14,4))
#plt.yscale("log")
#plt.xscale("log")
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()


#linkehälfte
NN1price = Sequential() 
NN1price.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1price.add(ZeroPadding2D(padding=(2, 2)))
NN1price.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1price.add(ZeroPadding2D(padding=(3,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,2)))
NN1price.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1price.add(MaxPooling2D(pool_size=(2, 1)))
#NN1price.add(Dropout(0.25))
#NN1price.add(ZeroPadding2D(padding=(0,1)))
NN1price.add(Conv2D(5, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1price.add(MaxPooling2D(pool_size=(4, 1)))
NN1price.summary()
#NN1price.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN1price.compile(loss ="MSE", optimizer = "adam",metrics=["MAPE","MSE"])


NN1price.fit(X_train_trafo, y_train_trafo1_price[:,:,:,[0,1,2,3,4]], batch_size=64, validation_data = (X_val_trafo, y_val_trafo1_price[:,:,:,[0,1,2,3,4]]), epochs = 50, verbose = True, shuffle=1)
#NN1c.save_weights("pricerweights_noriskfreerate_price_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
#NN1c.load_weights("pricerweights_noriskfreerate_price_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean

#  Results 
# The following plots show the performance on the testing set
S0=1.
#y_test_re    = yinversetransform(y_test_tmp,0).reshape((Ntest-1500,Nmaturities,Nstrikes))
#prediction   = NN1c.predict(X_test_tmp).reshape((Ntest-1500,Nmaturities,Nstrikes))

y_test_re    = yinversetransform(y_test_trafo1_price[:,:,:,[0,1,2,3,4]]).reshape((Ntest,Nmaturities,5))
prediction   = NN1price.predict(X_test_trafo).reshape((Ntest,Nmaturities,5))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
err_matrix = np.mean(err_rel_mat,axis=(1,2))
err_idx = np.argsort(err_matrix)
plt.figure(figsize= (14,4))
#plt.plot(np.min(y_test_re,axis=(1,2)))
#plt.plot(err_matrix)
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()



#rechte hälfte

NN1price = Sequential() 
NN1price.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1price.add(ZeroPadding2D(padding=(2, 2)))
NN1price.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1price.add(ZeroPadding2D(padding=(3,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,2)))
NN1price.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1price.add(MaxPooling2D(pool_size=(2, 1)))
#NN1price.add(Dropout(0.25))
#NN1price.add(ZeroPadding2D(padding=(0,1)))
NN1price.add(Conv2D(4, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1price.add(MaxPooling2D(pool_size=(4, 1)))
NN1price.summary()
NN1price.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NN1price.compile(loss ="MSE", optimizer = "adam",metrics=["MAPE","MSE"])


NN1price.fit(X_train_trafo, y_train_trafo1_price[:,:,:,[5,6,7,8]], batch_size=64, validation_data = (X_val_trafo, y_val_trafo1_price[:,:,:,[5,6,7,8]]), epochs = 50, verbose = True, shuffle=1)
#NN1c.save_weights("pricerweights_noriskfreerate_price_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
#NN1c.load_weights("pricerweights_noriskfreerate_price_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
y_test_re    = yinversetransform(y_test_trafo1_price[:,:,:,[5,6,7,8]]).reshape((Ntest,Nmaturities,4))
prediction   = NN1price.predict(X_test_trafo).reshape((Ntest,Nmaturities,4))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
err_matrix = np.mean(err_rel_mat,axis=(1,2))
err_idx = np.argsort(err_matrix)
plt.figure(figsize= (14,4))
#plt.plot(np.min(y_test_re,axis=(1,2)))
#plt.plot(err_matrix)
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.show()
#rrechts unten
NN1price = Sequential() 
NN1price.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1price.add(ZeroPadding2D(padding=(2, 0)))
NN1price.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1price.add(ZeroPadding2D(padding=(3,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(1,1)))
NN1price.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,2)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1price.add(ZeroPadding2D(padding=(2,1)))
NN1price.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1price.add(MaxPooling2D(pool_size=(2, 1)))
#NN1price.add(Dropout(0.25))
#NN1price.add(ZeroPadding2D(padding=(0,1)))
NN1price.add(Conv2D(4, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1price.add(MaxPooling2D(pool_size=(4, 1)))
NN1price.summary()
NN1price.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NN1price.compile(loss ="MSE", optimizer = "adam",metrics=["MAPE","MSE"])


NN1price.fit(X_train_trafo, 1000*y_train_trafo1_price[:,:,[5,6,7,8],[5,6,7,8]], batch_size=64, validation_data = (X_val_trafo, 1000*y_val_trafo1_price[:,:,[5,6,7,8],[5,6,7,8]]), epochs = 50, verbose = True, shuffle=1)
#NN1c.save_weights("pricerweights_noriskfreerate_price_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
#NN1c.load_weights("pricerweights_noriskfreerate_price_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
y_test_re    =1000*y_test_trafo1_price[:,:,[5,6,7,8],:]
y_test_re    =y_test_re[:,:,:,[5,6,7,8]]
prediction   = NN1price.predict(X_test_trafo).reshape((Ntest,4,4))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
err_matrix = np.mean(err_rel_mat,axis=(1,2))
err_idx = np.argsort(err_matrix)
plt.figure(figsize= (14,4))
#plt.plot(np.min(y_test_re,axis=(1,2)))
#plt.plot(err_matrix)
plt.xscale("log")
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())

plt.show()

# In[Test CNN as Decoder / MultiInputStructure]:

#model_comb = InputLayer(input_shape=(Nparameters))
#model_comb = Dense(30, activation="relu")(model_comb)
#model_comb = Dense(30, activation="relu")(model_comb)
#model_comb = Dense(Nparameters, activation="linear")(model_comb)
tf.config.experimental_run_functions_eagerly(False)
y_train_comb = np.concatenate((y_train_trafo2,y_train_trafo2_price),axis = 3)
y_test_comb = np.concatenate((y_test_trafo2,y_test_trafo2_price),axis = 3)
y_val_comb = np.concatenate((y_val_trafo2,y_val_trafo2_price),axis = 3)

y_train_comb1 = np.concatenate((y_train_trafo1,y_train_trafo1_price),axis = 1)
y_test_comb1 = np.concatenate((y_test_trafo1,y_test_trafo1_price),axis = 1)
y_val_comb1 = np.concatenate((y_val_trafo1,y_val_trafo1_price),axis = 1)
NN1comb = Sequential() 
NN1comb.add(InputLayer(input_shape=(Nparameters,)))
NN1comb.add(Dense(10,activation = "elu",use_bias=True))
NN1comb.add(Dense(30,activation = "elu",use_bias=True))
NN1comb.add(Dense(60,activation = "elu",use_bias=True))
NN1comb.add(Dense(120,activation = "elu",use_bias=True))
NN1comb.add(Reshape((15, 2,4), input_shape=(120,)))
NN1comb.add(ZeroPadding2D(padding=(2, 2)))
NN1comb.add(Conv2D(8, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1comb.add(ZeroPadding2D(padding=(1,1)))
NN1comb.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1comb.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1comb.add(ZeroPadding2D(padding=(1,1)))
NN1comb.add(Conv2D(9, (3,3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1comb.add(ZeroPadding2D(padding=(1,1)))
NN1comb.add(Conv2D(9, (3, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1comb.add(ZeroPadding2D(padding=(1,1)))
NN1comb.add(Conv2D(9, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1comb.add(ZeroPadding2D(padding=(1,1)))
NN1comb.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
#NN1comb.add(Conv2D(Nstrikes, (1, 1),padding='valid',use_bias =True,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1comb.summary()
#setting
NN1comb.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE"])
#NN1comb.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NN1comb.fit(X_train_trafo, y_train_trafo1, batch_size=64, validation_data = (X_val_trafo, y_val_trafo1), epochs = 300, verbose = True, shuffle=1)
NN1comb.fit(X_train_trafo.reshape((Ntrain,Nparameters)), y_train_comb1, batch_size=64, validation_data = (X_val_trafo.reshape((Nval,Nparameters)), y_val_comb1), epochs = 300, verbose = True, shuffle=1)
NN1comb.save_weights("pricerweights_comb.h5")
#NN1.load_weights("pricerweights_noriskfreerate.h5")

#  Results 
# The following plots show the performance on the testing set
S0=1.
y_test_re    = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = NN1.predict(X_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)



NN2comb = Sequential() 
NN2comb.add(InputLayer(input_shape=(Nmaturities,Nstrikes,2)))
NN2comb.add(Conv2D(12,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='elu'))
NN2comb.add(Conv2D(24,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2comb.add(MaxPooling2D(pool_size=(2, 2)))
NN2comb.add(Conv2D(32,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2comb.add(ZeroPadding2D(padding=(1,1)))
NN2comb.add(Conv2D(32,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2comb.add(ZeroPadding2D(padding=(1,1)))
NN2comb.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2comb.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2comb.add(ZeroPadding2D(padding=(1,1)))
NN2comb.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2comb.add(ZeroPadding2D(padding=(1,1)))
NN2comb.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2comb.add(Flatten())
NN2comb.add(Dense(Nparameters,activation = "tanh", kernel_constraint = tf.keras.constraints.NonNeg(),use_bias=True))
NN2comb.summary()
NN2comb.compile(loss =mse_constraint(0.25), optimizer = "adam",metrics=["MAPE", "MSE"])
#history = NN2comb.fit(y_train_comb,X_train_trafo2, batch_size=50, validation_data = (y_val_comb,X_val_trafo2), epochs=300, verbose = True, shuffle=1)
#NN2comb.save_weights("calibrationweights_comb.h5")
NN2comb.load_weights("calibrationweights_comb.h5")

from add_func_9x9 import calibration_plotter
prediction_calibration = NN2comb.predict(y_test_comb)
prediction_invtrafo= np.array([myinverse(x) for x in prediction_calibration])

#plots
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2 = calibration_plotter(prediction_calibration,X_test_trafo2,X_test)


# ### 3.2 Testing the performace of the AutoEncoder/Decoder Combination
# We test how the two previously trained NN work together. First, HNG-Vola surfaces are used to predict the underlying parameters with NN2. Those predictions are fed into NN1 to get Vola-Surface again. The results are shown below.

forecast = autoencoder(NN1comb,NN2comb)(y_test_comb)
#prediction = NN2.predict(y_test_trafo2)
#prediction_trafo = prediction.reshape((Ntest,Nparameters,1,1))
#forecast = NN1.predict(prediction_trafo).reshape(Ntest,Nmaturities,Nstrikes)
y_true_test = y_test_trafo2.reshape(Ntest,Nmaturities,Nstrikes)
mape_autoencoder,mse_autoencoder = plotter_autoencoder(forecast,y_true_test,y_test,testing_violation,testing_violation2)


"""



"""
# define two sets of inputs
inputA = Input(shape=(32,))
inputB = Input(shape=(128,))
# the first branch operates on the first input
x = Dense(8, activation="relu")(inputA)
x = Dense(4, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)
# the second branch opreates on the second input
y = Dense(64, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(4, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)
# combine the output of the two branches
combined = concatenate([x.output, y.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(2, activation="relu")(combined)
z = Dense(1, activation="linear")(z)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)
"""

# In[2.6 CNN as Encoder / Conv1D]:
"""
NN1e = Sequential() 
NN1e.add(InputLayer(input_shape=(Nparameters,1,)))
NN1e.add(ZeroPadding1D(padding=(1, 1)))
NN1e.add(Conv1D(4, 2, padding='valid',use_bias =True,strides =(1),activation='elu'))
NN1e.add(Conv1D(12, 2,padding='valid',use_bias =True,strides =(1),activation ='elu'))
NN1e.add(ZeroPadding1D(padding=(3,3)))
NN1e.add(Conv1D(24, 2,padding='valid',use_bias =True,strides =(1),activation ='elu'))
NN1e.add(Conv1D(36, 2,padding='valid',use_bias =True,strides =(1),activation ='elu'))
NN1e.add(ZeroPadding1D(padding=(3,3)))
NN1e.add(Conv1D(48, 2,padding='valid',use_bias =True,strides =(2),activation ='elu'))
NN1e.add(Conv1D(48, 2,padding='valid',use_bias =True,strides =(1),activation ='elu'))
NN1e.add(ZeroPadding1D(padding=(3,3)))
NN1e.add(Conv1D(36, 2,padding='valid',use_bias =True,strides =(2),activation ='elu'))
NN1e.add(Conv1D(24, 2,padding='valid',use_bias =True,strides =(1),activation ='elu'))
NN1e.add(ZeroPadding1D(padding=(2,2)))
NN1e.add(Conv1D(12, 2,padding='valid',use_bias =True,strides =(1),activation ='sigmoid'))
NN1e.add(Conv1D(12, 2,padding='valid',use_bias =True,strides =(1),activation ='sigmoid'))
NN1e.add(Conv1D(9, 1,padding='valid',use_bias =True,strides =(1),activation ='linear',kernel_constraint = tf.keras.constraints.NonNeg()))
NN1e.summary()

#setting
NN1e.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN1e.fit(X_train_trafo[:,:,:,0], y_train_trafo2.reshape((Ntrain,7,9)), batch_size=64, validation_data = (X_val_trafo[:,:,:,0], y_val_trafo2.reshape((Nval,7,9))),
        epochs = 70, verbose = True, shuffle=1)
#NN1e.save_weights("pricerweights_conv1d.h5")
NN1e.load_weights("pricerweights_conv1d.h5")

#  Results 
# The following plots show the performance on the testing set
S0=1.
y_test_re    = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = NN1e.predict(X_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)

"""















# In[3.1 CNN as  Decoder/Inverse Mapping / Calibration]:

NN2 = Sequential() 
NN2.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(MaxPooling2D(pool_size=(2, 2)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(Flatten())
NN2.add(Dense(Nparameters,activation = 'linear',use_bias=True))
NN2.summary()
#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])

#setting
NN2.compile(loss =mse_constraint(0.75), optimizer = "adam",metrics=["MAPE", "MSE"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#history = NN2.fit(y_train_trafo2_price,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2_price,X_val_trafo2), epochs=40, verbose = True, shuffle=1,callbacks =[es])
#NN2.save_weights("calibrationweights_price.h5")#id_3283354135d44b67_data_price_norm_231046clean
NN2.load_weights("calibrationweights_price.h5")#id_3283354135d44b67_data_price_norm_231046clean



# ### 3.1 Results

from add_func_9x9 import calibration_plotter
prediction_calibration = NN2.predict(y_test_trafo2)
prediction_invtrafo= np.array([myinverse(x) for x in prediction_calibration])

#plots
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2 = calibration_plotter(prediction_calibration,X_test_trafo2,X_test)


NN2a = Sequential() 
NN2a.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2a.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(MaxPooling2D(pool_size=(2, 2)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(Flatten())
NN2a.add(Dense(3,activation = 'linear',use_bias=True))
NN2a.summary()
#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])

#setting
NN2a.compile(loss =mse_constraint(0.75), optimizer = "adam",metrics=["MAPE", "MSE"])
#history = NN2a.fit(y_train_trafo2_price,X_train_trafo2[:,[0,1,2]], batch_size=50, validation_data = (y_val_trafo2_price,X_val_trafo2[:,[0,1,2]]), epochs=40, verbose = True, shuffle=1)
#NN2a.save_weights("calibrationweights_a_price.h5")#id_3283354135d44b67_data_price_norm_231046clean
NN2a.load_weights("calibrationweights_a_price.h5")#id_3283354135d44b67_data_price_norm_231046clean



# ### 3.1 Results

prediction_calibration_a = NN2a.predict(y_test_trafo2)


NN2b = Sequential() 
NN2b.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2b.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='tanh'))
NN2b.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2b.add(MaxPooling2D(pool_size=(2, 2)))
NN2b.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2b.add(ZeroPadding2D(padding=(1,1)))
NN2b.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2b.add(ZeroPadding2D(padding=(1,1)))
NN2b.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2b.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2b.add(ZeroPadding2D(padding=(1,1)))
NN2b.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2b.add(ZeroPadding2D(padding=(1,1)))
NN2b.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2b.add(Flatten())
NN2b.add(Dense(2,activation = 'linear',use_bias=True))
NN2b.summary()
#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])

#setting
NN2b.compile(loss ="MSE", optimizer = "adam",metrics=["MAPE", "MSE"])
#history = NN2b.fit(y_train_trafo2_price,X_train_trafo2[:,[3,4]], batch_size=50, validation_data = (y_val_trafo2_price,X_val_trafo2[:,[3,4]]), epochs=40, verbose = True, shuffle=1)
#NN2b.save_weights("calibrationweights_b_price.h5")#id_3283354135d44b67_data_price_norm_231046clean
NN2b.load_weights("calibrationweights_b_price.h5")#id_3283354135d44b67_data_price_norm_231046clean



# ### 3.1 Results

from add_func_9x9 import calibration_plotter
prediction_calibration_b = NN2b.predict(y_test_trafo2)
prediction_calibration = np.concatenate((prediction_calibration_a,prediction_calibration_b),axis=1)

prediction_invtrafo= np.array([myinverse(x) for x in prediction_calibration])

#plots
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2 = calibration_plotter(prediction_calibration,X_test_trafo2,X_test)


























































# ### 3.2 Testing the performace of the AutoEncoder/Decoder Combination
# We test how the two previously trained NN work together. First, HNG-Vola surfaces are used to predict the underlying parameters with NN2. Those predictions are fed into NN1 to get Vola-Surface again. The results are shown below.

#forecast = autoencoder(NN1,NN2)(y_test_trafo2)
#prediction = NN2.predict(y_test_trafo2)
#prediction_trafo = prediction.reshape((Ntest,Nparameters,1,1))
#forecast = NN1.predict(prediction_trafo).reshape(Ntest,Nmaturities,Nstrikes)
y_true_test = y_test_trafo2.reshape(Ntest,Nmaturities,Nstrikes)
mape_autoencoder,mse_autoencoder = plotter_autoencoder(forecast,y_true_test,y_test,testing_violation,testing_violation2)
# In[3.1 CNN as  Decoder/Inverse Mapping / Prices]:

NN2a = Sequential() 
NN2a.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2a.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(MaxPooling2D(pool_size=(2, 2)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(Flatten())
NN2a.add(Dense(Nparameters,activation = 'linear',use_bias=True))
NN2a.summary()
#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])

#setting
NN2a.compile(loss =mse_constraint(0.75), optimizer = "adam",metrics=["MAPE", "MSE"])
history = NN2a.fit(y_train_trafo2_price,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2_price,X_val_trafo2), epochs=40, verbose = True, shuffle=1)
NN2.save_weights("calibrationweights_price_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
#NN2a.load_weights("calibrationweights_pric_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean



# ### 3.1 Results

from add_func_9x9 import calibration_plotter
prediction_calibration = NN2a.predict(y_test_trafo2_price)
prediction_invtrafo= np.array([myinverse(x) for x in prediction_calibration])

#plots
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2 = calibration_plotter(prediction_calibration,X_test_trafo2,X_test)


# ### 3.2 Testing the performace of the AutoEncoder/Decoder Combination
# We test how the two previously trained NN work together. First, HNG-Vola surfaces are used to predict the underlying parameters with NN2. Those predictions are fed into NN1 to get Vola-Surface again. The results are shown below.

forecast = autoencoder(NN1,NN2)(y_test_trafo2)
#prediction = NN2.predict(y_test_trafo2)
#prediction_trafo = prediction.reshape((Ntest,Nparameters,1,1))
#forecast = NN1.predict(prediction_trafo).reshape(Ntest,Nmaturities,Nstrikes)
y_true_test = y_test_trafo2.reshape(Ntest,Nmaturities,Nstrikes)
mape_autoencoder,mse_autoencoder = plotter_autoencoder(forecast,y_true_test,y_test,testing_violation,testing_violation2)


# In[3.2 CNN as  Decoder/Inverse Mapping / Calibration TANHELU]:

NN2c = Sequential() 
NN2c.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2c.add(Conv2D(12,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='elu'))
NN2c.add(Conv2D(24,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2c.add(MaxPooling2D(pool_size=(2, 2)))
NN2c.add(Conv2D(32,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2c.add(ZeroPadding2D(padding=(1,1)))
NN2c.add(Conv2D(32,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2c.add(ZeroPadding2D(padding=(1,1)))
NN2c.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2c.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2c.add(ZeroPadding2D(padding=(1,1)))
NN2c.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2c.add(ZeroPadding2D(padding=(1,1)))
NN2c.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='elu'))
NN2c.add(Flatten())
NN2c.add(Dense(Nparameters,activation = "tanh", kernel_constraint = tf.keras.constraints.NonNeg(),use_bias=True))
NN2c.summary()

#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])

#setting
NN2c.compile(loss =mse_constraint(0.75), optimizer = "adam",metrics=["MAPE", "MSE"])
history = NN2c.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2), epochs=40, verbose = True, shuffle=1)
NN2.save_weights("calibrationweights_elutanh.h5")
NN2c.load_weights("calibrationweights_elutanh.h5")


# 5.1 Results


from add_func_9x9 import calibration_plotter
prediction_calibration = NN2c.predict(y_test_trafo2)
prediction_invtrafo= np.array([myinverse(x) for x in prediction_calibration])

#plots
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2 = calibration_plotter(prediction_calibration,X_test_trafo2,X_test)

# In[4.1 Testing the performace of the AutoEncoder/Decoder Combination]
# We test how the two previously trained NN work together. First, HNG-Vola surfaces are used to predict the underlying parameters with NN2. Those predictions are fed into NN1 to get Vola-Surface again. The results are shown below.


forecast = autoencoder(NN1,NN2c)(y_test_trafo2)
#prediction = NN2.predict(y_test_trafo2)
#prediction_trafo = prediction.reshape((Ntest,Nparameters,1,1))
#forecast = NN1.predict(prediction_trafo).reshape(Ntest,Nmaturities,Nstrikes)
y_true_test = y_test_trafo2.reshape(Ntest,Nmaturities,Nstrikes)
mape_autoencoder,mse_autoencoder = plotter_autoencoder(forecast,y_true_test,y_test,testing_violation,testing_violation2)




















"""TEST"""
# In[2.1 TESTING CNN as Encoder / Pricing Kernel with noriskfree rate]:
NN1t = Sequential() 
NN1t.add(InputLayer(input_shape=(Nparameters,)))
NN1t.add(Dense(10,activation = "elu",use_bias=True))
NN1t.add(Dense(30,activation = "elu",use_bias=True))
NN1t.add(Dense(60,activation = "elu",use_bias=True))
NN1t.add(Dense(120,activation = "elu",use_bias=True))
NN1t.add(Reshape((15, 2,4), input_shape=(120,)))
NN1t.add(ZeroPadding2D(padding=(1, 2)))
NN1t.add(Conv2D(32, (3, 3), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1t.add(ZeroPadding2D(padding=(1,1)))
NN1t.add(Conv2D(32, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1t.add(Conv2D(32, (3, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1t.add(ZeroPadding2D(padding=(1,2)))
NN1t.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1t.add(ZeroPadding2D(padding=(1,2)))
NN1t.add(Conv2D(32, (3, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1t.add(ZeroPadding2D(padding=(1,1)))
NN1t.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
NN1t.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1t.summary()
#setting
#NN1.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE"])
NN1t.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN1t.fit(X_train_trafo2, y_train_trafo1, batch_size=64, validation_data = (X_val_trafo2, y_val_trafo1), epochs = 30, verbose = True, shuffle=1)
#NN1t.save_weights("pricerweights_noriskfreerate.h5")
#NN1t.load_weights("pricerweights_noriskfreerate.h5")

#  Results 
# The following plots show the performance on the testing set
S0=1.
y_test_re    = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = NN1t.predict(X_test_trafo2).reshape((Ntest,Nmaturities,Nstrikes))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
# In[2.1 TESTING 2 CNN as Encoder / Pricing Kernel with noriskfree rate]:
NN1t = Sequential() 
NN1t.add(InputLayer(input_shape=(Nparameters,)))
NN1t.add(Dense(10,activation = "elu",use_bias=True))
NN1t.add(Dense(30,activation = "elu",use_bias=True))
NN1t.add(Dense(60,activation = "elu",use_bias=True))
NN1t.add(Dense(120,activation = "elu",use_bias=True))
NN1t.add(Dense(240,activation = "elu",use_bias=True))
NN1t.add(Reshape((15, 4,4), input_shape=(240,)))
NN1t.add(ZeroPadding2D(padding=(1, 1)))
NN1t.add(Conv2D(32, (3, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1t.add(ZeroPadding2D(padding=(1,2)))
NN1t.add(Conv2D(32, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1t.add(Conv2D(32, (3, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1t.add(ZeroPadding2D(padding=(2,2)))
NN1t.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1t.add(ZeroPadding2D(padding=(1,1)))
NN1t.add(Conv2D(32, (3, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1t.add(ZeroPadding2D(padding=(1,1)))
NN1t.add(Conv2D(32, (3, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
NN1t.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1t.summary()
#setting
#NN1t.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE"])
NN1t.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN1t.fit(X_train_trafo2, y_train_trafo1, batch_size=64, validation_data = (X_val_trafo2, y_val_trafo1), epochs = 30, verbose = True, shuffle=1)
#NN1t.save_weights("pricerweights_noriskfreerate.h5")
#NN1t.load_weights("pricerweights_noriskfreerate.h5")

#  Results 
# The following plots show the performance on the testing set
S0=1.
y_test_re    = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = NN1t.predict(X_test_trafo2).reshape((Ntest,Nmaturities,Nstrikes))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)


