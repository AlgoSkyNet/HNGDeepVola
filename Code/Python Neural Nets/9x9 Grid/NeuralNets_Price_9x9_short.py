# In[Preambel]:
import numpy as np
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.compat.v1.keras.layers import Reshape,InputLayer,Dense,Flatten, Conv2D,Conv1D, Dropout, Input,ZeroPadding2D,ZeroPadding1D,MaxPooling2D
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 0 , 'CPU': 20} ) 
#sess = tf.compat.v1.Session(config=config) 
#tf.compat.v1.keras.backend.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tf.compat.v1.keras.backend.set_floatx('float64')  

## import parameter data set
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

# In[PricingNetwork Sigmoid Activation RelMSE Train]:
inputs_train =np.concatenate((X_train_trafo,rates_train.reshape((Ntrain,Nmaturities,1,1))),axis=1)
inputs_val = np.concatenate((X_val_trafo,rates_val.reshape((Nval,Nmaturities,1,1))),axis=1)
inputs_test = np.concatenate((X_test_trafo,rates_test.reshape((Ntest,Nmaturities,1,1))),axis=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 30 ,restore_best_weights=True)

### architecture
NNprice_sig2 = Sequential() 
NNprice_sig2.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_sig2.add(ZeroPadding2D(padding=(2, 2)))
NNprice_sig2.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig2.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sig2.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig2.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig2.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sig2.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig2.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NNprice_sig2.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig2.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(1,1),activation ='elu'))
NNprice_sig2.add(Conv2D(4, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNprice_sig2.summary()

### trainingsetting
NNprice_sig2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NNprice_sig2.fit(inputs_train, y_train_trafo1_price[:,:,:,[5,6,7,8]], batch_size=64, validation_data = (inputs_val, y_val_trafo1_price[:,:,:,[5,6,7,8]]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNprice_sig2.save_weights("price_weights_rate_9x9_sig2.h5")
NNprice_sig2.load_weights("price_weights_rate_9x9_sig2.h5")



NNprice_sig1 = Sequential() 
NNprice_sig1.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_sig1.add(ZeroPadding2D(padding=(2, 2)))
NNprice_sig1.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig1.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sig1.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig1.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig1.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sig1.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig1.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NNprice_sig1.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig1.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(1,1),activation ='elu'))
NNprice_sig1.add(Conv2D(5, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNprice_sig1.summary()

#setting
NNprice_sig1.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NNprice_sig1.fit(inputs_train, y_train_trafo1_price[:,:,:,[0,1,2,3,4]], batch_size=64, validation_data = (inputs_val, y_val_trafo1_price[:,:,:,[0,1,2,3,4]]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNprice_sig1.save_weights("priceweights_rates_9x9_sig1.h5")
NNprice_sig1.load_weights("priceweights_rates_9x9_sig1.h5")




NNprice_sig3 = Sequential() 
NNprice_sig3.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_sig3.add(ZeroPadding2D(padding=(2, 1)))
NNprice_sig3.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig3.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sig3.add(ZeroPadding2D(padding=(2,1)))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig3.add(ZeroPadding2D(padding=(2,2)))
NNprice_sig3.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sig3.add(ZeroPadding2D(padding=(2,1)))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig3.add(ZeroPadding2D(padding=(2,1)))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NNprice_sig3.add(ZeroPadding2D(padding=(2,1)))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sig3.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(1,1),activation ='elu'))
NNprice_sig3.add(Conv2D(9, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
NNprice_sig3.summary()

#setting
NNprice_sig3.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NNprice_sig3.fit(inputs_train, y_train_trafo1_price[:,:,[0,8],:], batch_size=64, validation_data = (inputs_val, y_val_trafo1_price[:,:,[0,8],:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNprice_sig3.save_weights("priceweights_rates_9x9_sig3.h5")
NNprice_sig3.load_weights("priceweights_rates_9x9_sig3.h5")


###  Results 
prediction_sig2   = NNprice_sig2.predict(inputs_test).reshape((Ntest,Nmaturities,4))
prediction_sig1   = NNprice_sig1.predict(inputs_test).reshape((Ntest,Nmaturities,5))
prediction_sig3   = NNprice_sig3.predict(inputs_test).reshape((Ntest,2,Nstrikes))
prediction_full_sig = np.concatenate((prediction_sig1,prediction_sig2),axis = 2)
prediction_full_sig[:,[0,8],:] = prediction_sig3
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))

err_rel_mat_sig,tmp,tmp,tmp = pricing_plotter(prediction_full_sig,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes)))
err_matrix_sig = np.mean(err_rel_mat_sig,axis=(1,2))
sig_mape_median = np.median(err_rel_mat_sig,axis=0)
sig_mape_mean = np.mean(err_rel_mat_sig,axis=0)
sig_mape_max = np.max(err_rel_mat_sig,axis =0)

plt.figure(figsize= (14,4))
#plt.xscale("log")
#plt.yscale("log")
plt.hist(err_rel_mat_sig.flatten(),bins=100)
plt.show()
plt.figure(figsize= (14,4))
#plt.xscale("log")
#plt.yscale("log")
plt.hist(err_matrix_sig.flatten(),bins=100)
plt.show()
#plt.figure(figsize= (14,4))
#plt.yscale("log")
#plt.xscale("log")
#plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
#plt.show()

#from matplotlib.colors import LogNorm
#plt.figure(figsize= (14,4))
#ax = plt.subplot(1,1,1)
#plt.imshow(100*sig_mape_mean,norm=LogNorm(vmin=100*sig_mape_mean.min(), vmax=100*sig_mape_mean.max()))
#plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
#plt.colorbar(format=mtick.PercentFormatter())
#ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
#ax.set_xticklabels(strikes)
#ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
#ax.set_yticklabels(maturities)
#plt.xlabel("Strike",fontsize=15,labelpad=5)
#plt.ylabel("Maturity",fontsize=15,labelpad=5)
#plt.show()
fig = plt.figure()
for i in range(Nmaturities):
    for j in range(Nstrikes):
        plt.subplot(Nmaturities,Nstrikes,Nmaturities*i+j+1)
        plt.hist(y_test_re[:,i,j].flatten(),bins=100)
for ax in fig.get_axes():
    ax.label_outer()
plt.show
fig = plt.figure()
for i in range(Nmaturities):
    for j in range(Nstrikes):
        plt.subplot(Nmaturities,Nstrikes,Nmaturities*i+j+1)
        plt.hist(err_rel_mat_sig[:,i,j].flatten(),bins=100)
for ax in fig.get_axes():
    ax.label_outer()
plt.show()

# In[PricingNetwork Linear Activation RelMSE Train]:

### architecture
NNprice_lin2 = Sequential() 
NNprice_lin2.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_lin2.add(ZeroPadding2D(padding=(2, 2)))
NNprice_lin2.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin2.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_lin2.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin2.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin2.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_lin2.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin2.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NNprice_lin2.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin2.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(1,1),activation ='elu'))
NNprice_lin2.add(Conv2D(4, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNprice_lin2.summary()

### trainingsetting
NNprice_lin2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NNprice_lin2.fit(inputs_train, y_train_trafo1_price[:,:,:,[5,6,7,8]], batch_size=64, validation_data = (inputs_val, y_val_trafo1_price[:,:,:,[5,6,7,8]]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNprice_lin2.save_weights("price_weights_rate_9x9_linear2.h5")
NNprice_lin2.load_weights("price_weights_rate_9x9_linear2.h5")



NNprice_lin1 = Sequential() 
NNprice_lin1.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_lin1.add(ZeroPadding2D(padding=(2, 2)))
NNprice_lin1.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin1.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_lin1.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin1.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin1.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_lin1.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin1.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NNprice_lin1.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin1.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(1,1),activation ='elu'))
NNprice_lin1.add(Conv2D(5, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNprice_lin1.summary()

#setting
NNprice_lin1.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NNprice_lin1.fit(inputs_train, y_train_trafo1_price[:,:,:,[0,1,2,3,4]], batch_size=64, validation_data = (inputs_val, y_val_trafo1_price[:,:,:,[0,1,2,3,4]]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNprice_lin1.save_weights("priceweights_rates_9x9_linear.h5")
NNprice_lin1.load_weights("priceweights_rates_9x9_linear.h5")

NNprice_lin3 = Sequential() 
NNprice_lin3.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_lin3.add(ZeroPadding2D(padding=(2, 1)))
NNprice_lin3.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin3.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_lin3.add(ZeroPadding2D(padding=(2,1)))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin3.add(ZeroPadding2D(padding=(2,2)))
NNprice_lin3.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_lin3.add(ZeroPadding2D(padding=(2,1)))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin3.add(ZeroPadding2D(padding=(2,1)))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NNprice_lin3.add(ZeroPadding2D(padding=(2,1)))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_lin3.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(1,1),activation ='elu'))
NNprice_lin3.add(Conv2D(9, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
NNprice_lin3.summary()

#setting
NNprice_lin3.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NNprice_lin3.fit(inputs_train, y_train_trafo1_price[:,:,[0,8],:], batch_size=64, validation_data = (inputs_val, y_val_trafo1_price[:,:,[0,8],:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNprice_lin3.save_weights("priceweights_rates_9x9_linear3.h5")
NNprice_lin3.load_weights("priceweights_rates_9x9_linear3.h5")


###  Results 
prediction_lin2   = NNprice_lin2.predict(inputs_test).reshape((Ntest,Nmaturities,4))
prediction_lin1   = NNprice_lin1.predict(inputs_test).reshape((Ntest,Nmaturities,5))
prediction_lin3   = NNprice_lin3.predict(inputs_test).reshape((Ntest,2,Nstrikes))
prediction_full_lin = np.concatenate((prediction_lin1,prediction_lin2),axis = 2)
prediction_full_lin[:,[0,8],:] = prediction_lin3
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))

err_rel_mat_lin,tmp,tmp,tmp = pricing_plotter(prediction_full_lin,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes)))
err_matrix_lin = np.mean(err_rel_mat_lin,axis=(1,2))
lin_mape_median = np.median(err_rel_mat_lin,axis=0)
lin_mape_mean = np.mean(err_rel_mat_lin,axis=0)
lin_mape_max = np.max(err_rel_mat_lin,axis =0)

plt.figure(figsize= (14,4))
#plt.xscale("log")
#plt.yscale("log")
plt.hist(err_rel_mat_lin.flatten(),bins=100)
plt.show()
plt.figure(figsize= (14,4))
#plt.xscale("log")
#plt.yscale("log")
plt.hist(err_matrix_lin.flatten(),bins=100)
plt.show()
#plt.figure(figsize= (14,4))
#plt.yscale("log")
#plt.xscale("log")
#plt.scatter(y_test_re.flatten(),err_rel_mat.flatten())
#plt.show()

#from matplotlib.colors import LogNorm
#plt.figure(figsize= (14,4))
#ax = plt.subplot(1,1,1)
#plt.imshow(100*lin_mape_mean,norm=LogNorm(vmin=100*lin_mape_mean.min(), vmax=100*lin_mape_mean.max()))
#plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
#plt.colorbar(format=mtick.PercentFormatter())
#ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
#ax.set_xticklabels(strikes)
#ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
#ax.set_yticklabels(maturities)
#plt.xlabel("Strike",fontsize=15,labelpad=5)
#plt.ylabel("Maturity",fontsize=15,labelpad=5)
#plt.show()

fig = plt.figure()
for i in range(Nmaturities):
    for j in range(Nstrikes):
        plt.subplot(Nmaturities,Nstrikes,Nmaturities*i+j+1)
        plt.hist(y_test_re[:,i,j].flatten(),bins=100)
for ax in fig.get_axes():
    ax.label_outer()
plt.show
fig = plt.figure()
for i in range(Nmaturities):
    for j in range(Nstrikes):
        plt.subplot(Nmaturities,Nstrikes,Nmaturities*i+j+1)
        plt.hist(err_rel_mat_lin[:,i,j].flatten(),bins=100)
for ax in fig.get_axes():
    ax.label_outer()
plt.show()

# In[3.1 CNN as  Decoder/Inverse Mapping / Calibration]:
from add_func_9x9 import calibration_plotter

#SCALE PRICES BEFORE!!!!!
y_train_price_scale = ytransform(y_train_price,0).reshape((Ntrain,Nmaturities,Nstrikes,1))
y_test_price_scale = ytransform(y_test_price,0).reshape((Ntest,Nmaturities,Nstrikes,1))
y_val_price_scale = ytransform(y_val_price,0).reshape((Nval,Nmaturities,Nstrikes,1))
NN2s = Sequential() 
NN2s.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2s.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='tanh'))
NN2s.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2s.add(MaxPooling2D(pool_size=(2, 2)))
NN2s.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2s.add(ZeroPadding2D(padding=(1,1)))
NN2s.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2s.add(ZeroPadding2D(padding=(1,1)))
NN2s.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2s.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2s.add(ZeroPadding2D(padding=(1,1)))
NN2s.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2s.add(ZeroPadding2D(padding=(1,1)))
NN2s.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2s.add(Flatten())
NN2s.add(Dense(Nparameters,activation = 'linear',use_bias=True,kernel_constraint = tf.keras.constraints.NonNeg()))
NN2s.summary()
#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
from add_func_9x9 import log_constraint,miss_count,mape_constraint
#setting
#NN2.compile(loss =mse_constraint(0.75), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
NN2s.compile(loss =log_constraint(param=1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
es = EarlyStopping(monitor='val_MSE', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
history = NN2s.fit(y_train_price_scale,X_train_trafo2, batch_size=120, validation_data = (y_val_price_scale,X_val_trafo2), epochs=100, verbose = True, shuffle=1,callbacks =[es])
NN2s.compile(loss =log_constraint(param=0.01,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
NN2s.save_weights("calibrationweights_price_scale.h5")
#NN2s.load_weights("calibrationweights_price_scale.h5")

es = EarlyStopping(monitor='val_MAPE', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
history = NN2s.fit(y_train_price_scale,X_train_trafo2, batch_size=120, validation_data = (y_val_price_scale,X_val_trafo2), epochs=100, verbose = True, shuffle=1,callbacks =[es])
NN2s.save_weights("calibrationweights_price_scale2.h5")
#NN2s.load_weights("calibrationweights_price_scale.h5")


prediction_calibration = NN2s.predict(y_test_price_scale)
prediction_invtrafo= np.array([myinverse(x) for x in prediction_calibration])

#plots
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2 = calibration_plotter(prediction_calibration,X_test_trafo2,X_test)










