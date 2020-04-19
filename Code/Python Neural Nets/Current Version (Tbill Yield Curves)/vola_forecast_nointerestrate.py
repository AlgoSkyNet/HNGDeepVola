# Vola Forecasting no riskrate adjustments
"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import keras
import tensorflow as tf
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess
"""
# In[1. Initialisation/ Preambel and Data Import]:
# This Initialisation will be used for everyfile to ensure the same conditions everytime!
def names_data():
    name_price = "id_3283354135d44b67_data_price_norm_231046clean.mat"
    name_vola = "id_3283354135d44b67_data_vola_norm_231046clean.mat"
    return name_price,name_vola


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.compat.v1.keras.layers import Reshape,InputLayer,Dense,Flatten, Conv2D,Conv1D, Dropout, Input,ZeroPadding2D,ZeroPadding1D,MaxPooling2D
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

## import data set
from config_latent import Nparameters,maturities,strikes,Nstrikes,Nmaturities,Ntest,Ntrain,Nval
from config_latent import xx,rates_train,rates_val,rates_test,ub,lb,diff,bound_sum
from config_latent import X_train,X_test,X_val,X_train_trafo,X_val_trafo,X_test_trafo,X_train_trafo2,X_val_trafo2,X_test_trafo2
# vola
from config_latent import yy,y_train,y_test,y_val,ub_vola,lb_vola,diff_vola,bound_sum_vola
from config_latent import y_train_trafo,y_val_trafo,y_test_trafo
from config_latent import y_train_trafo1,y_val_trafo1,y_test_trafo1
from config_latent import y_train_trafo2,y_val_trafo2,y_test_trafo2
# price
from config_latent import yy_price,y_train_price,y_test_price,y_val_price,ub_price,lb_price,diff_price,bound_sum_price
from config_latent import y_train_trafo_price,y_val_trafo_price,y_test_trafo_price
from config_latent import y_train_trafo1_price,y_val_trafo1_price,y_test_trafo1_price
from config_latent import y_train_trafo2_price,y_val_trafo2_price,y_test_trafo2_price

# import custom functions #scaling tools
from config_latent import ytransform, yinversetransform,myscale,myinverse

#custom errors
from add_func_latent import root_mean_squared_error,root_relative_mean_squared_error,mse_constraint,rmse_constraint
#else
from add_func_latent import constraint_violation,pricing_plotter,calibration_plotter_deterministic,plotter_autoencoder

tf.compat.v1.keras.backend.set_floatx('float64')  


def autoencoder(nn1,nn2):
    def autoencoder_predict(y_values):
        prediction = nn2.predict(y_values)
        prediction_trafo = prediction.reshape((Ntest,Nparameters,1,1))
        forecast = nn1.predict(prediction_trafo).reshape(Ntest,Nmaturities,Nstrikes)
        return forecast
    return autoencoder_predict

## In[2.1 CNN as Encoder / Vola Kernel with no riskfree rate]:

NN1 = Sequential() 
NN1.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1.add(ZeroPadding2D(padding=(2, 2)))
NN1.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
NN1.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1.summary()
#setting
#NN1.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE"])
NN1.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#good num of epoch could be 350-(Ntotal/1000)
#NN1.fit(X_train_trafo, y_train_trafo1, batch_size=64, validation_data = (X_val_trafo, y_val_trafo1), epochs = 100, verbose = True, shuffle=1)
#NN1.save_weights("pricerweights_noriskfreerate_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
NN1.load_weights("pricerweights_noriskfreerate_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean

#  Results 
# The following plots show the performance on the testing set
S0=1.
y_test_re    = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = NN1.predict(X_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
plt.figure(figsize=(14,4))
plt.hist(err_rel_mat.flatten(),log=True,density =True)
plt.figure(figsize=(14,4))
plt.hist(err_rel_mat.flatten(),density =True)
plt.figure(figsize=(14,4))
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.figure(figsize=(14,4))
plt.hist((prediction.flatten(),y_test_re.flatten()))

NN1 = Sequential() 
NN1.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1.add(ZeroPadding2D(padding=(2, 2)))
NN1.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
NN1.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1.summary()
#setting
#NN1.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE"])
NN1.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#good num of epoch could be 350-(Ntotal/1000)
NN1.fit(X_train_trafo, y_train_trafo1, batch_size=64, validation_data = (X_val_trafo, y_val_trafo1), epochs = 100, verbose = True, shuffle=1)
#NN1.save_weights("pricerweights_noriskfreerate_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
#NN1.load_weights("pricerweights_noriskfreerate_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean

#  Results 
# The following plots show the performance on the testing set
S0=1.
y_test_re    = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = NN1.predict(X_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
plt.figure(figsize=(14,4))
plt.hist(err_rel_mat.flatten(),log=True,density =True)
plt.figure(figsize=(14,4))
plt.hist(err_rel_mat.flatten(),density =True)
plt.figure(figsize=(14,4))
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.figure(figsize=(14,4))
plt.hist((prediction.flatten(),y_test_re.flatten()))
NN1 = Sequential() 
NN1.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1.add(ZeroPadding2D(padding=(2, 2)))
NN1.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
NN1.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='relu', kernel_constraint = tf.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1.summary()
#setting
#NN1.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE"])
NN1.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#good num of epoch could be 350-(Ntotal/1000)
NN1.fit(X_train_trafo, y_train_trafo1, batch_size=64, validation_data = (X_val_trafo, y_val_trafo1), epochs = 100, verbose = True, shuffle=1)
#NN1.save_weights("pricerweights_noriskfreerate_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
#NN1.load_weights("pricerweights_noriskfreerate_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean

#  Results 
# The following plots show the performance on the testing set
S0=1.
y_test_re    = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = NN1.predict(X_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
#plots
err_rel_mat,err_mat,idx,bad_idx = pricing_plotter(prediction,y_test_re)
plt.figure(figsize=(14,4))
plt.hist(err_rel_mat.flatten(),log=True,bins=100)
plt.figure(figsize=(14,4))
plt.hist(err_rel_mat.flatten(),bins=100)
plt.figure(figsize=(14,4))
plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
plt.figure(figsize=(14,4))
plt.hist((prediction.flatten(),y_test_re.flatten()))

















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
NN2.add(Dense(Nparameters,activation = 'linear',use_bias=True,kernel_constraint=tf.keras.constraints.UnitNorm(axis=0)))
NN2.summary()
#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])

#setting
#NN2.compile(loss =mse_constraint(0.75), optimizer = "adam",metrics=["MAPE", "MSE"])
#history = NN2.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2), epochs=40, verbose = True, shuffle=1)
#NN2.save_weights("calibrationweights_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean
NN2.load_weights("calibrationweights_231046.h5")#id_3283354135d44b67_data_price_norm_231046clean



# ### 3.1 Results

from add_func_latent import calibration_plotter
prediction_calibration = NN2.predict(y_test_trafo2)
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
