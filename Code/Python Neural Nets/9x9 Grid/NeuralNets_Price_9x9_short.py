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
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 0 , 'CPU': 64} ) 
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
K.set_floatx('float64')  

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

err_rel_mat_sig,err_mse_mat_sig,err_optt_mat_sig,err_iv_mat_sig,tmp,tmp = pricing_plotter(prediction_full_sig,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes)))
err_matrix_sig = np.mean(err_rel_mat_sig,axis=(1,2))
sig_mape_median = np.median(err_rel_mat_sig,axis=0)
sig_mape_mean = np.mean(err_rel_mat_sig,axis=0)
sig_mape_max = np.max(err_rel_mat_sig,axis =0)

plt.figure(figsize= (14,4))
plt.title("Empirical Relative Error Distribution")
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("Rel Error")
plt.ylabel("Count")
plt.hist(err_rel_mat_sig.flatten(),bins=100)
plt.show()
plt.figure(figsize= (14,4))
plt.title("Empirical Relative Error Distributrion per Surface")
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("Mean Relative Error per Surface")
plt.ylabel("Count")
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
plt.suptitle("Distribution of True Prices per Gridpoint")
for i in range(Nmaturities):
    for j in range(Nstrikes):
        plt.subplot(Nmaturities,Nstrikes,Nmaturities*i+j+1)
        plt.hist(y_test_re[:,i,j].flatten(),bins=100)
        plt.xlim([0,0.31])
for ax in fig.get_axes():
    ax.label_outer()
plt.show()
fig = plt.figure()
plt.suptitle("Distribution of Relative Errors per Gridpoint")

for i in range(Nmaturities):
    for j in range(Nstrikes):
        plt.subplot(Nmaturities,Nstrikes,Nmaturities*i+j+1)
        plt.hist(err_rel_mat_sig[:,i,j].flatten(),bins=100)
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
#es = EarlyStopping(monitor='val_MSE', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
#history = NN2s.fit(y_train_price_scale,X_train_trafo2, batch_size=120, validation_data = (y_val_price_scale,X_val_trafo2), epochs=100, verbose = True, shuffle=1,callbacks =[es])
#NN2s.save_weights("calibrationweights_price_scale.h5")
NN2s.load_weights("calibrationweights_price_scale.h5")

prediction_calibration1 = NN2s.predict(y_test_price_scale)
prediction_invtrafo1= np.array([myinverse(x) for x in prediction_calibration1])

#plots
error_cal1,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp = calibration_plotter(prediction_calibration1,X_test_trafo2,X_test)




NN2s.compile(loss =log_constraint(param=0.01,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
#es = EarlyStopping(monitor='val_MAPE', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
#history = NN2s.fit(y_train_price_scale,X_train_trafo2, batch_size=120, validation_data = (y_val_price_scale,X_val_trafo2), epochs=100, verbose = True, shuffle=1,callbacks =[es])
#NN2s.save_weights("calibrationweights_price_scale2.h5")
NN2s.load_weights("calibrationweights_price_scale.h5")


prediction_calibration2 = NN2s.predict(y_test_price_scale)
prediction_invtrafo2= np.array([myinverse(x) for x in prediction_calibration2])

#plots
error_cal2,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp = calibration_plotter(prediction_calibration2,X_test_trafo2,X_test)



# In[NEWMODEL]:
# In[PricingNetwork Sigmoid Activation RelMSE Train]:
inputs_train =np.concatenate((X_train_trafo,rates_train.reshape((Ntrain,Nmaturities,1,1))),axis=1)
inputs_val = np.concatenate((X_val_trafo,rates_val.reshape((Nval,Nmaturities,1,1))),axis=1)
inputs_test = np.concatenate((X_test_trafo,rates_test.reshape((Ntest,Nmaturities,1,1))),axis=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 30 ,restore_best_weights=True)

bad_test    = np.min(vega_test, axis=1) <1e-5    
bad_train   = np.min(vega_train, axis=1) <1e-5    
bad_val     = np.min(vega_val, axis=1) <1e-5    
good_test   = np.min(vega_test, axis=1) >= 1e-5    
good_train  = np.min(vega_train, axis=1) >= 1e-5    
good_val    = np.min(vega_val, axis=1) >= 1e-5  
n_valg = np.sum(good_val)
n_testg = np.sum(good_test)
n_traing = np.sum(good_train)
n_valb = Nval-np.sum(good_val)
n_testb = Ntest-np.sum(good_test)
n_trainb = Ntrain-np.sum(good_train)


"""
NNprice_sigNEW1 = Sequential() 
NNprice_sigNEW1.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_sigNEW1.add(ZeroPadding2D(padding=(2, 1)))
NNprice_sigNEW1.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW1.add(ZeroPadding2D(padding=(2,2)))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sigNEW1.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW1.add(ZeroPadding2D(padding=(2,2)))
NNprice_sigNEW1.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sigNEW1.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW1.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NNprice_sigNEW1.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_sigNEW1.add(Conv2D(9, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation =sig_scaled, kernel_constraint = tf.keras.constraints.NonNeg()))
NNprice_sigNEW1.summary()
NNprice_sigNEW2 = Sequential() 
NNprice_sigNEW2.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_sigNEW2.add(ZeroPadding2D(padding=(2, 1)))
NNprice_sigNEW2.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW2.add(ZeroPadding2D(padding=(2,2)))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sigNEW2.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW2.add(ZeroPadding2D(padding=(2,2)))
NNprice_sigNEW2.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sigNEW2.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW2.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NNprice_sigNEW2.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_sigNEW2.add(Conv2D(9, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation =sig_scaled, kernel_constraint = tf.keras.constraints.NonNeg()))
NNprice_sigNEW2.summary()
NNprice_sigNEW3 = Sequential() 
NNprice_sigNEW3.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_sigNEW3.add(ZeroPadding2D(padding=(2, 1)))
NNprice_sigNEW3.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW3.add(ZeroPadding2D(padding=(2,2)))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sigNEW3.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW3.add(ZeroPadding2D(padding=(2,2)))
NNprice_sigNEW3.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNprice_sigNEW3.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW3.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =False,strides =(2,1),activation ='elu'))
NNprice_sigNEW3.add(ZeroPadding2D(padding=(2,1)))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_sigNEW3.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_sigNEW3.add(Conv2D(9, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation =sig_scaled, kernel_constraint = tf.keras.constraints.NonNeg()))
NNprice_sigNEW3.summary()
y_train_trafo1_price_good = y_train_trafo1_price[good_train,:,:,:]
y_test_trafo1_price_good = y_test_trafo1_price[good_test,:,:,:]
y_val_trafo1_price_good = y_val_trafo1_price[good_val,:,:,:]

#setting
NNprice_sigNEW1.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNprice_sigNEW1.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price_good[:,:,[0,1,2],:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], 2000*y_val_trafo1_price_good[:,:,[0,1,2],:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNprice_sigNEW2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNprice_sigNEW2.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price_good[:,:,[3,4,5],:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], 2000*y_val_trafo1_price_good[:,:,[3,4,5],:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNprice_sigNEW3.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNprice_sigNEW3.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price_good[:,:,[6,7,8],:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], 2000*y_val_trafo1_price_good[:,:,[6,7,8],:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNprice_sigNEW1.save_weights("price_relmse_weights_3neta_2000.h5")
NNprice_sigNEW2.save_weights("price_relmse_weights_3netb_2000.h5")
NNprice_sigNEW3.save_weights("price_relmse_weights_3netc_2000.h5")

#NNprice_sigNEW1.save_weights("price_relmse_weights_3neta.h5")
#NNprice_sigNEW2.save_weights("price_relmse_weights_3netb.h5")
#NNprice_sigNEW3.save_weights("price_relmse_weights_3netc.h5")

###  Results 
prediction_sigNEW1   = NNprice_sigNEW1.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,3,Nmaturities))
prediction_sigNEW2   = NNprice_sigNEW2.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,3,Nmaturities))
prediction_sigNEW3   = NNprice_sigNEW3.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,3,Nmaturities))

prediction_full_sigNEW = np.concatenate((prediction_sigNEW1,prediction_sigNEW2,prediction_sigNEW3),axis = 1)
y_test_re    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))
err_rel_mat_full,err_mse_mat_full,err_opt_mat_full,err_iv_mat_full,tmp,tmp = pricing_plotter(prediction_full_sigNEW,y_test_re[good_test,:,:],2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])
err_matrix_sig = np.mean(err_rel_mat_full,axis=(1,2))
full_mape_median = np.median(err_rel_mat_full,axis=0)
full_mape_mean = np.mean(err_rel_mat_full,axis=0)
full_mape_max = np.max(err_rel_mat_full,axis =0)


plt.figure()
plt.subplot(2,1,1)
plt.plot(err_iv_mat_full[:,0,0])

plt.subplot(2,1,2)
plt.plot(vega_test1[:,0,0,0])
plt.show()

    """

# In[2.3 CNN as Encoder / Pricing Kernel with riskfree rate]:
def sig_scaled(a,b,c):
    def sig_tmp(x):
        return a / (1 + K.exp(-b*(x-c)))
    return sig_tmp
def ivrmse_approx(y_true_with_vega, y_pred):
    return K.sqrt(K.mean(K.square((y_pred - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:])))

def ivrmse(vega):
    def ivrmse_tmp(y_true, y_pred):
            return K.sqrt(K.mean(K.square((y_pred - y_true)/vega)))
    return ivrmse_tmp

def option_log_likelyhood(y_true_with_vega, y_pred):
        return K.mean(K.log(K.square((y_pred - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:])))



NNpriceFULL = Sequential() 
NNpriceFULL.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNpriceFULL.add(ZeroPadding2D(padding=(2, 2)))
NNpriceFULL.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='relu'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL.summary()


#NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#NNpriceFULL.fit(inputs_train, y_train_trafo1_price, batch_size=64, validation_data = (inputs_val, y_val_trafo1_price), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
vega_train1 = np.asarray([vega_train[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
vega_test1 = np.asarray([vega_test[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])
vega_val1 = np.asarray([vega_val[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])

y_train_tmp = np.concatenate((2000*y_train_trafo1_price,2000*vega_train1),axis=1) 
y_test_tmp  = np.concatenate((2000*y_test_trafo1_price,2000*vega_test1),axis=1)
y_val_tmp   = np.concatenate((2000*y_val_trafo1_price,2000*vega_val1),axis=1)
bad_test    = np.min(vega_test, axis=1) <1e-5    
bad_train   = np.min(vega_train, axis=1) <1e-5    
bad_val     = np.min(vega_val, axis=1) <1e-5    
good_test   = np.min(vega_test, axis=1) >= 1e-5    
good_train  = np.min(vega_train, axis=1) >= 1e-5    
good_val    = np.min(vega_val, axis=1) >= 1e-5  
n_valg = np.sum(good_val)
n_testg = np.sum(good_test)
n_traing = np.sum(good_train)
n_valb = Nval-np.sum(good_val)
n_testb = Ntest-np.sum(good_test)
n_trainb = Ntrain-np.sum(good_train)



def ivrmse_approx_no11(y_true_with_vega, y_pred):
    return K.sqrt(81/80*K.mean(K.square((y_pred[:,0,:,:] - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:]))-1/81*K.square((y_pred[:,0,0,0] - y_true_with_vega[:,0,0,0])/y_true_with_vega[:,1,0,0]))
    
def new_norm(y_true_with_vega,y_pred):
    return K.mean(K.norm(K.square((y_pred[:,0,:,:] - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:]),ord="euclidian",dim=(1,2)))
    



#2000
NNpriceFULL.compile(loss = ivrmse_approx, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL.fit(inputs_train[good_train,:,:,:], y_train_tmp[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], y_val_tmp[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_ivrmse_weights_1net_2000.h5")
NNpriceFULL.load_weights("price_ivrmse_weights_1net_2000.h5")

NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:],  2000*y_val_trafo1_price[good_val,:,:,:]), epochs =10, verbose = True, shuffle=1,callbacks=[es])

NNpriceFULL.compile(loss = ivrmse_approx_no11, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
#NNpriceFULL.fit(inputs_train[good_train,:,:,:], y_train_tmp[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], y_val_tmp[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_ivrmse11_weights_1net_2000.h5")
NNpriceFULL.load_weights("price_ivrmse11_weights_1net_2000.h5")


#NNpriceFULL.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE","MSE"])
#NNpriceFULL.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], 2000*y_val_trafo1_price[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL.save_weights("price_mse_weights_1net.h5")

prediction_iv_test_g   = NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_g    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_iv_test_g,y_test_re_g,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])
mean11_mape = np.mean(err_rel_mat,axis=0)
mean11_mse = np.mean(err_mat,axis=0)
mean11_optll = np.mean(err_optll,axis=0)
mean1_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))

###  Results 
prediction_iv_test   = NNpriceFULL.predict(inputs_test).reshape((Ntest,Nmaturities,Nstrikes))
prediction_iv_val   = NNpriceFULL.predict(inputs_val).reshape((Nval,Nmaturities,Nstrikes))
prediction_iv_train   = NNpriceFULL.predict(inputs_train).reshape((Ntrain,Nmaturities,Nstrikes))

y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))
y_train_re    = yinversetransform(y_train_trafo_price).reshape((Ntrain,Nmaturities,Nstrikes))
y_val_re    = yinversetransform(y_val_trafo_price).reshape((Nval,Nmaturities,Nstrikes))


tmp,tmp,tmp,ivtest,tmp,tmp = pricing_plotter(prediction_iv_test,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes)))
tmp,tmp,tmp,ivtrain,tmp,tmp = pricing_plotter(prediction_iv_train,y_train_re,vega_train.reshape((Ntrain,Nmaturities,Nstrikes)))
tmp,tmp,tmp,ivval,tmp,tmp = pricing_plotter(prediction_iv_val,y_val_re,vega_val.reshape((Nval,Nmaturities,Nstrikes)))

plt.figure()
plt.subplot(3,1,1)
plt.plot(ivtest[:,0,0])
plt.subplot(3,1,2)
plt.plot(ivtrain[:,0,0])
plt.subplot(3,1,3)
plt.plot(ivval[:,0,0])

#plt.subplot(2,1,2)
#plt.plot(vega_test1[:,0,0,0])
plt.show()
#idx_approx = np.argsort(err_iv_mat_full[:,0,0], axis=None)
    
plt.figure()
plt.subplot(3,1,1)
plt.xscale("log")
plt.scatter(vega_test1[good_test,:,:,:].flatten(),ivtest.flatten())
plt.subplot(3,1,2)
plt.xscale("log")
plt.scatter(vega_train1[:,0,0,0],ivtrain[:,0,0])
plt.subplot(3,1,3)
plt.xscale("log")
plt.scatter(vega_val1[:,0,0,0],ivval[:,0,0])
plt.show()

plt.figure()
plt.subplot(3,1,1)
plt.hist(vega_test1[:,0,0,0],bins=100)
plt.subplot(3,1,2)
plt.hist(vega_val1[:,0,0,0],bins=100)
plt.subplot(3,1,3)
plt.hist(vega_val1[:,0,0,0],bins=100)
plt.show()

# base1
NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL.fit(inputs_train[:,:,:,:], y_train_trafo1_price[:,:,:,:], batch_size=64, validation_data = (inputs_val[:,:,:,:], y_val_trafo1_price[:,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_relmse_weights_1net.h5")
NNpriceFULL.load_weights("price_relmse_weights_1net.h5")
prediction_relmse_base1   = NNpriceFULL.predict(inputs_test[:,:,:,:]).reshape((Ntest,Nmaturities,Nstrikes))
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))
tmp,tmp,tmp,ivtest,tmp,tmp = pricing_plotter(prediction_relmse_base1,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes)))

#base1 vega cleaned
NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL.fit(inputs_train[good_train,:,:,:], y_train_trafo1_price[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], y_val_trafo1_price[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_relmse_weights_1net_g.h5")
NNpriceFULL.load_weights("price_relmse_weights_1net_g.h5")
prediction_relmse_base1_g   = NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_1g    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
tmp,tmp,tmp,ivtest4,tmp,tmp = pricing_plotter(prediction_relmse_base1_g,y_test_re_1g,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])

# 2000 base
#NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
#NNpriceFULL.fit(inputs_train[:,:,:,:], 2000*y_train_trafo1_price[:,:,:,:], batch_size=64, validation_data = (inputs_val[:,:,:,:],2000*y_val_trafo1_price[:,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL.save_weights("price_relmse_weights_1net_2000.h5")
NNpriceFULL.load_weights("price_relmse_weights_1net_2000.h5")
prediction_relmse_base2000   = NNpriceFULL.predict(inputs_test[:,:,:,:]).reshape((Ntest,Nmaturities,Nstrikes))
y_test_re    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_relmse_base2000,y_test_re,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes)))

#base2000 vega cleaned
NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:],2000*y_val_trafo1_price[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_relmse_weights_1net_2000g.h5")
NNpriceFULL.load_weights("price_relmse_weights_1net_2000g.h5")
prediction_relmse_base1_2000g   = NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_1g    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
tmp,tmp,tmp,ivtest4,tmp,tmp = pricing_plotter(prediction_relmse_base1_2000g,y_test_re_1g,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])



#base1 vega cleaned and scaled
train_scale = ytransform(y_train_trafo1_price,2)
val_scale = ytransform(y_val_trafo1_price,2)
test_scale = ytransform(y_test_trafo1_price,2)

NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL.fit(inputs_train[good_train,:,:,:], train_scale[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:],val_scale[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_relmse_weights_1net_scale.h5")
NNpriceFULL.load_weights("price_relmse_weights_1net_scale.h5")
prediction_relmse_base1_scale   = yinversetransform(NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes)),2)
y_test_re_1g    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
tmp,tmp,tmp,ivtest4,tmp,tmp = pricing_plotter(prediction_relmse_base1_g,y_test_re_1g,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])


#base1 vega cleaned relu
NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL.fit(inputs_train[good_train,:,:,:], y_train_trafo1_price[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], y_val_trafo1_price[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_relmse_weights_1net_g_relu.h5")
NNpriceFULL.load_weights("price_relmse_weights_1net_g_relu.h5")
prediction_relmse_base1_grelu  = NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_1g    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_relmse_base1_grelu,y_test_re_1g,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])

#base1 vega cleaned transpose
train_t = np.asarray([y_train_trafo1_price[i,0,:,:].T for i in range(Ntrain)]).reshape((Ntrain,9,9,1))
test_t = np.asarray([y_test_trafo1_price[i,0,:,:].T for i in range(Ntest)]).reshape((Ntest,9,9,1))
val_t = np.asarray([y_val_trafo1_price[i,0,:,:].T for i in range(Nval)]).reshape((Nval,9,9,1))

NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL.fit(inputs_train[good_train,:,:,:], train_t[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], val_t[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_relmse_weights_1net_g_Trans.h5")
NNpriceFULL.load_weights("price_relmse_weights_1net_g_Trans.h5")
prediction_relmse_base1_t   = NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re   = test_t[good_test,:,:]
vega_t =np.asarray([vega_test1[i,0,:,:].T for i in range(Nval)]).reshape((Nval,9,9,1))
tmp,tmp,tmp,ivtest4,tmp,tmp = pricing_plotter(prediction_relmse_base1_t,y_test_re,vega_t[good_test,:,:])







# In[Testing Area Zero Padding Structure: NOT SOLVING THE PROBLEM]
"""
NNpriceFULL2 = Sequential() 
NNpriceFULL2.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNpriceFULL2.add(ZeroPadding2D(padding=(5, 7)))
NNpriceFULL2.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
#NNpriceFULL2.add(Conv2D(1, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL2.add(Conv2D(9, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='sigmoid'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL2.summary()

NNpriceFULL2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL2.fit(inputs_train[:,:,:,:], y_train_trafo2_price[:,:,:,:].reshape((Ntrain,9,1,9)), batch_size=64, validation_data = (inputs_val[:,:,:,:], y_val_trafo2_price[:,:,:,:].reshape((Nval,9,1,9))), epochs =150, verbose = True, shuffle=1,callbacks=[es])
prediction_zeropad   = NNpriceFULL2.predict(inputs_test[:,:,:,:]).reshape((Ntest,Nmaturities,Nstrikes))
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp = pricing_plotter(prediction_zeropad,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes)))
mean2_mape = np.mean(err_rel_mat,axis=0)
mean2_mse = np.mean(err_mat,axis=0)
mean2_optll = np.mean(err_optll,axis=0)
mean2_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))
"""
"""
NNpriceFULL2 = Sequential() 
NNpriceFULL2.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNpriceFULL2.add(ZeroPadding2D(padding=(5, 7)))
NNpriceFULL2.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL2.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
#NNpriceFULL2.add(Conv2D(1, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL2.add(Conv2D(9, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='sigmoid'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL2.summary()

y_train_tmp2 = np.concatenate((y_train_trafo1_price.reshape((Ntrain,9,1,9)),vega_train1.reshape((Ntrain,9,1,9))),axis=2) 
y_test_tmp2  = np.concatenate((y_test_trafo1_price.reshape((Ntest,9,1,9)),vega_test1.reshape((Ntest,9,1,9))),axis=2)
y_val_tmp2   = np.concatenate((y_val_trafo1_price.reshape((Nval,9,1,9)),vega_val1.reshape((Nval,9,1,9))),axis=2)
def ivrmse_approx2(y_true_with_vega, y_pred):
    return K.sqrt(K.mean(K.square((y_pred[:,:,0,:] - y_true_with_vega[:,:,0,:])/y_true_with_vega[:,:,1,:])))

#NNpriceFULL2.compile(loss = ivrmse_approx2, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
#NNpriceFULL2.fit(inputs_train[good_train,:,:,:], y_train_tmp2[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], y_val_tmp2[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL2.save_weights("price_relmse_weights_1net_pad.h5")
NNpriceFULL2.load_weights("price_relmse_weights_1net_pad.h5")

prediction_zeropad2   = NNpriceFULL2.predict(inputs_test[:,:,:,:]).reshape((Ntest,Nmaturities,Nstrikes))
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_zeropad,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes)))
mean22_mape = np.mean(err_rel_mat,axis=0)
mean22_mse = np.mean(err_mat,axis=0)
mean22_optll = np.mean(err_optll,axis=0)
mean22_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))
"""




# In[Testing Area: CONV1]
"""
NNpriceFULL3 = Sequential() 
NNpriceFULL3.add(InputLayer(input_shape=(Nparameters+Nmaturities,1)))
NNpriceFULL3.add(ZeroPadding1D(padding=(4)))
NNpriceFULL3.add(Conv1D(32, (2), padding='valid',use_bias =True,strides =(1),activation='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL3.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
#NNpriceFULL3.add(Conv1D(1, (2, 2),padding='valid',use_bias =True,strides =(1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL3.add(Conv1D(9, (2),padding='valid',use_bias =True,strides =(1),activation ='sigmoid'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL3.summary()

# 1D Construct
#NNpriceFULL3.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
#NNpriceFULL3.fit(inputs_train[good_train,:,:,0], y_train_trafo1_price[good_train,0,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], y_val_trafo1_price[good_val,0,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL3.save_weights("price_relmse_weights_1net_1D.h5")
NNpriceFULL3.load_weights("price_relmse_weights_1net_1D.h5")
prediction_1D   = NNpriceFULL3.predict(inputs_test[good_test,:,:,0])
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_1D,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])

NNpriceFULL4 = Sequential() 
NNpriceFULL4.add(InputLayer(input_shape=(Nparameters+Nmaturities,1)))
NNpriceFULL4.add(ZeroPadding1D(padding=(6)))
NNpriceFULL4.add(Conv1D(32, (2), padding='valid',use_bias =True,strides =(1),activation='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL4.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
#NNpriceFULL4.add(Conv1D(1, (2, 2),padding='valid',use_bias =True,strides =(1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL4.add(Conv1D(9, (2),padding='valid',use_bias =True,strides =(1),activation ='sigmoid'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL4.add(Reshape((1,9,9)))
NNpriceFULL4.summary()

# 1D Construct more weights (no reshape)
NNpriceFULL4.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL4.fit(inputs_train[good_train,:,:,0], y_train_trafo1_price[good_train,0,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], y_val_trafo1_price[good_val,0,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL4.save_weights("price_relmse_weights_1net_1D2.h5")
NNpriceFULL4.load_weights("price_relmse_weights_1net_1D2.h5")
prediction_1D2   = NNpriceFULL4.predict(inputs_test[good_test,:,:,0])
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_1D2,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])

#IVRMSE opt (reshape)
def ivrmse_approx2(y_true_with_vega, y_pred):
    return K.sqrt(K.mean(K.square((y_pred[:,0,:,:] - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:])))
y_train_full = np.concatenate((y_train_trafo1_price,vega_train1),axis=1) 
y_test_full  = np.concatenate((y_test_trafo1_price,vega_test1),axis=1)
y_val_full  = np.concatenate((y_val_trafo1_price,vega_val1),axis=1)
NNpriceFULL4.compile(loss = ivrmse_approx2, optimizer = "adam")
NNpriceFULL4.fit(inputs_train[good_train,:,:,0], y_train_full[good_train,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], y_val_full[good_val,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL4.save_weights("price_ivrmse_weights_1net_1D2.h5")
NNpriceFULL4.load_weights("price_ivrmse_weights_1net_1D2.h5")
prediction_1D3   = NNpriceFULL4.predict(inputs_test[good_test,:,:,0])[:,0,:,:]
prediction_1D3train   = NNpriceFULL4.predict(inputs_train[good_train,:,:,0])[:,0,:,:]
prediction_1D3val  = NNpriceFULL4.predict(inputs_val[good_val,:,:,0])[:,0,:,:]

std_pred = np.std(prediction_1D3 ,axis=0)
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_1D3,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])



NNpriceFULL5 = Sequential() 
NNpriceFULL5.add(InputLayer(input_shape=(Nparameters+Nmaturities,1)))
NNpriceFULL5.add(ZeroPadding1D(padding=(7)))
NNpriceFULL5.add(Conv1D(32, (2), padding='valid',use_bias =True,strides =(1),activation='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL5.add(Conv1D(32, (2),padding='valid', use_bias =True,strides =(1),activation ='elu'))
#NNpriceFULL5.add(Conv1D(1, (2, 2),padding='valid',use_bias =True,strides =(1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL5.add(Conv1D(9, (2),padding='valid',use_bias =True,strides =(1),activation ='sigmoid'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL5.summary()

# 1D Construct more weights (no reshape)
NNpriceFULL5.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL5.fit(inputs_train[good_train,:,:,0], y_train_trafo1_price[good_train,0,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], y_val_trafo1_price[good_val,0,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL5.save_weights("price_relmse_weights_1net_1D5.h5")
NNpriceFULL5.load_weights("price_relmse_weights_1net_1D5.h5")
prediction_1D5   = NNpriceFULL5.predict(inputs_test[good_test,:,:,0])
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_1D5,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])
"""

"""
# 1D Construct more weights (no reshape)
NNpriceFULL6 = Sequential() 
NNpriceFULL6.add(InputLayer(input_shape=(Nparameters+Nmaturities,1)))
NNpriceFULL6.add(ZeroPadding1D(padding=(9)))
NNpriceFULL6.add(Conv1D(32, (2), padding='valid',use_bias =True,strides =(1),activation='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6.add(Conv1D(32, (2),padding='valid', use_bias =True,strides =(1),activation ='elu'))
#NNpriceFULL6.add(Conv1D(1, (2, 2),padding='valid',use_bias =True,strides =(1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL6.add(Conv1D(9, (2),padding='valid',use_bias =True,strides =(1),activation ='sigmoid'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL6.summary()
def relative_mean_squared_error(y_true, y_pred):
        return K.mean(K.square((y_pred - y_true)/y_true))   

NNpriceFULL6.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
history = NNpriceFULL6.fit(inputs_train[good_train,:,:,0], y_train_trafo1_price[good_train,0,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], y_val_trafo1_price[good_val,0,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL6.save_weights("price_relmse_weights_1net_1D6.h5")
NNpriceFULL6.load_weights("price_relmse_weights_1net_1D6.h5")
prediction_1D6   = NNpriceFULL6.predict(inputs_test[good_test,:,:,0])
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_1D6,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])

#even bigger vegas
good_test2   = np.min(vega_test, axis=1) >= 1e-4    
good_train2  = np.min(vega_train, axis=1) >= 1e-4    
good_val2    = np.min(vega_val, axis=1) >= 1e-4  
n_valg2 = np.sum(good_val2)
n_testg2 = np.sum(good_test2)
n_traing2 = np.sum(good_train2)
n_valb2 = Nval-np.sum(good_val2)
n_testb2 = Ntest-np.sum(good_test2)
n_trainb2 = Ntrain-np.sum(good_train2)
"""
"""
# 1D Construct more weights (no reshape)
NNpriceFULL7 = Sequential() 
NNpriceFULL7.add(InputLayer(input_shape=(Nparameters+Nmaturities,1)))
NNpriceFULL7.add(ZeroPadding1D(padding=(9)))
NNpriceFULL7.add(Conv1D(32, (2), padding='valid',use_bias =True,strides =(1),activation='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL7.add(Conv1D(32, (2),padding='valid', use_bias =True,strides =(1),activation ='elu'))
#NNpriceFULL7.add(Conv1D(1, (2, 2),padding='valid',use_bias =True,strides =(1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL7.add(Conv1D(9, (2),padding='valid',use_bias =True,strides =(1),activation ='sigmoid'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL7.summary()
from tensorflow.compat.v1.keras.optimizers import Adam
def relative_mean_squared_error(y_true, y_pred):
        return K.mean(K.square((y_pred - y_true)/y_true))   

NNpriceFULL7.compile(loss = root_relative_mean_squared_error, optimizer = Adam(clipvalue =0.1,clipnorm=0.1),metrics=["MAPE",root_mean_squared_error])
history = NNpriceFULL7.fit(inputs_train[good_train2,:,:,0], y_train_trafo1_price[good_train2,0,:,:], batch_size=64, validation_data = (inputs_val[good_val2,:,:,0], y_val_trafo1_price[good_val2,0,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL7.save_weights("price_relmse_weights_1net_1D7.h5")
NNpriceFULL7.load_weights("price_relmse_weights_1net_1D7.h5")
prediction_1D7   = NNpriceFULL7.predict(inputs_test[good_test2,:,:,0])
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test2,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_1D6,y_test_re,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test2,:,:])
"""
#structure 6 but with 2000
# 1D Construct more weights (no reshape)
NNpriceFULL6a = Sequential() 
NNpriceFULL6a.add(InputLayer(input_shape=(Nparameters+Nmaturities,1)))
NNpriceFULL6a.add(ZeroPadding1D(padding=(9)))
NNpriceFULL6a.add(Conv1D(32, (2), padding='valid',use_bias =True,strides =(1),activation='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(32, (2),padding='valid', use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6a.add(Conv1D(9, (2),padding='valid',use_bias =True,strides =(1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL6a.summary()
def relative_mean_squared_error(y_true, y_pred):
        return K.mean(K.square((y_pred - y_true)/y_true))   
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 30 ,restore_best_weights=True)

#NNpriceFULL6a.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
#history = NNpriceFULL6a.fit(inputs_train[good_train,:,:,0], 2000*y_train_trafo1_price[good_train,0,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], 2000*y_val_trafo1_price[good_val,0,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL6a.save_weights("price_relmse_weights_1net_1D6a.h5")
NNpriceFULL6a.load_weights("price_relmse_weights_1net_1D6a.h5")
prediction_1D6a   = NNpriceFULL6a.predict(inputs_test[good_test,:,:,0])
y_test_re    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_1D6a,y_test_re,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])
mean6a_mape = np.mean(err_rel_mat,axis=0)
mean6a_mse = np.mean(err_mat,axis=0)
mean6a_optll = np.mean(err_optll,axis=0)
mean6a_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))


"""
NNpriceFULL6a.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
#history = NNpriceFULL6a.fit(inputs_train[good_train,:,:,0], 2000*y_train_trafo1_price[good_train,0,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], 2000*y_val_trafo1_price[good_val,0,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NMNpriceFULL6a.save_weights("price_relmse_weights_1net_1D6a_MSE.h5")
NNpriceFULL6a.load_weights("price_relmse_weights_1net_1D6a_MSE.h5")
prediction_1D6amse   = NNpriceFULL6a.predict(inputs_test[good_test,:,:,0])

"""
y_test_re    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_1D6a,y_test_re,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])

dict_iv ={"price" : y_test_re,"forecast" : prediction_1D6a , "vega": 2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:], "param" : X_test[good_test,:],"rates": rates_test[good_test,:] }
scipy.io.savemat('data_for_IV.mat', dict_iv)


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 250 ,restore_best_weights=True)
NNpriceFULL6a.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE",root_mean_squared_error])
historylong = NNpriceFULL6a.fit(inputs_train[good_train,:,:,0], 2000*y_train_trafo1_price[good_train,0,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], 2000*y_val_trafo1_price[good_val,0,:,:]), epochs =1250, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL6a.save_weights("price_relmse_weights_1net_1D6a_LONG.h5")

# OPTIMISED PARAMETERS

name_price_tmp = "MLE_calib_price.mat"
name_vola_tmp = "MLE_calib_vola.mat"
name_vega_tmp = "MLE_calib_vega.mat"

path = "C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Python Neural Nets/9x9 Grid/Dataset/"
tmp         = scipy.io.loadmat(path+name_vola_tmp)
data_vola_tmp        =tmp['data_vola']
tmp         = scipy.io.loadmat(path+name_price_tmp)
data_price_tmp       = tmp['data_price']
tmp         = scipy.io.loadmat(path+name_vega_tmp)
data_vega_tmp       = tmp['data_vega'].reshape((459,9,9))
X_tmp = data_vola_tmp[:,:5]
X_tmp_trafo = np.array([myscale(x) for x in X_tmp])
rates_tmp = data_vola_tmp[:,5:14]
y_vola_tmp = data_vola_tmp[:,14:].reshape((459,9,9))
y_price_tmp = data_price_tmp[:,14:].reshape((459,9,9))

prediction_tmp   = NNpriceFULL6a.predict(np.concatenate((X_tmp_trafo,rates_tmp),axis=1).reshape(459,14,1))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_tmp,y_price_tmp,data_vega_tmp)
dict_iv_tmp ={"price" : y_price_tmp,"forecast" : prediction_tmp , "vega": data_vega_tmp, "param" : X_tmp,"rates": rates_tmp }
scipy.io.savemat('data_forecast_calibated.mat', dict_iv_tmp)


#testing with high interest rates

name_price_tmp = "id_a43c01784a9d43e3_data_price_norm_9881_bigprice.mat"
name_vola_tmp = "id_a43c01784a9d43e3_data_vola_norm_9881_bigprice.mat"
name_vega_tmp = "id_a43c01784a9d43e3_data_vega_norm_9881_bigprice.mat"

path = "C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Python Neural Nets/9x9 Grid/Dataset/"
tmp         = scipy.io.loadmat(path+name_vola_tmp)
data_vola_tmp        =tmp['data_vola']
tmp         = scipy.io.loadmat(path+name_price_tmp)
data_price_tmp       = tmp['data_price']
tmp         = scipy.io.loadmat(path+name_vega_tmp)
data_vega_tmp       = tmp['data_vega'].reshape((9881,9,9))
X_tmp = data_vola_tmp[:,:5]
X_tmp_trafo = np.array([myscale(x) for x in X_tmp])
rates_tmp = data_vola_tmp[:,5:14]
y_vola_tmp = data_vola_tmp[:,14:].reshape((9881,9,9))
y_price_tmp = data_price_tmp[:,14:].reshape((9881,9,9))

prediction_tmp   = NNpriceFULL6a.predict(np.concatenate((X_tmp_trafo,rates_tmp),axis=1).reshape(9881,14,1))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_tmp,y_price_tmp,data_vega_tmp)


"""
file:///C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Dataset Generator/id_991a3d750d864e8d_data_vola_norm_139540_bigprice.mat
file:///C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Dataset Generator/id_991a3d750d864e8d_data_price_norm_139540_bigprice.mat
file:///C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Dataset Generator/id_991a3d750d864e8d_data_vega_norm_139540_bigprice.mat
""""


name_price_tmp = "id_991a3d750d864e8d_data_price_norm_139540_bigprice.mat"
name_vola_tmp  = "id_991a3d750d864e8d_data_vola_norm_139540_bigprice.mat"
name_vega_tmp  = "id_991a3d750d864e8d_data_vega_norm_139540_bigprice.mat"

path = "C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Python Neural Nets/9x9 Grid/Dataset/"
tmp         = scipy.io.loadmat(path+name_vola_tmp)
data_vola_tmp        =tmp['data_vola']
tmp         = scipy.io.loadmat(path+name_price_tmp)
data_price_tmp       = tmp['data_price']
tmp         = scipy.io.loadmat(path+name_vega_tmp)
data_vega_tmp       = tmp['data_vega'].reshape((139540,9,9))
X_tmp = data_vola_tmp[:,:5]
X_tmp_trafo = np.array([myscale(x) for x in X_tmp])
rates_tmp = data_vola_tmp[:,5:14]
y_vola_tmp = data_vola_tmp[:,14:].reshape((139540,9,9))
y_price_tmp = data_price_tmp[:,14:].reshape((139540,9,9))

prediction_tmp   = NNpriceFULL6a.predict(np.concatenate((X_tmp_trafo,rates_tmp),axis=1).reshape((139540,14,1)))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_tmp,y_price_tmp,data_vega_tmp)
























"""

#IVRMSE opt (reshape)
NNpriceFULL6b = Sequential() 
NNpriceFULL6b.add(InputLayer(input_shape=(Nparameters+Nmaturities,1)))
NNpriceFULL6b.add(ZeroPadding1D(padding=(9)))
NNpriceFULL6b.add(Conv1D(32, (2), padding='valid',use_bias =True,strides =(1),activation='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(32, (2),padding='valid', use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6b.add(Conv1D(9, (2),padding='valid',use_bias =True,strides =(1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL6b.add(Reshape((1,9,9)))
NNpriceFULL6b.summary()
def ivrmse_approx2(y_true_with_vega, y_pred):
    return K.sqrt(K.mean(K.square((y_pred[:,0,:,:] - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:])))
y_train_full = 2000*np.concatenate((y_train_trafo1_price,vega_train1),axis=1) 
y_test_full  = 2000*np.concatenate((y_test_trafo1_price,vega_test1),axis=1)
y_val_full  = 2000*np.concatenate((y_val_trafo1_price,vega_val1),axis=1)

#NNpriceFULL6b.compile(loss = ivrmse_approx2, optimizer = Adam(clipvalue =0.1,clipnorm=0.1))
#NNpriceFULL6b.fit(inputs_train[good_train,:,:,0], y_train_full[good_train,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], y_val_full[good_val,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL6b.save_weights("price_ivrmse_weights_1net_1D6b.h5")
NNpriceFULL6b.load_weights("price_ivrmse_weights_1net_1D6b.h5")

NNpriceFULL6b.compile(loss = ivrmse_approx_no11, optimizer ="adam")# Adam(clipvalue =0.1,clipnorm=0.1))
NNpriceFULL6b.fit(inputs_train[good_train,:,:,0], y_train_full[good_train,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], y_val_full[good_val,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL6b.save_weights("price_ivrmse11_weights_1net_1D6b.h5")
NNpriceFULL6b.load_weights("price_ivrmse11_weights_1net_1D6b.h5")

prediction_1D6b   = NNpriceFULL4.predict(inputs_test[good_test,:,:,0])[:,0,:,:]
y_test_re    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_1D6b,2000*y_test_re,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])
mean6b_mape = np.mean(err_rel_mat,axis=0)
mean6b_mse = np.mean(err_mat,axis=0)
mean6b_optll = np.mean(err_optll,axis=0)
mean6b_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))


"""










# 


def sig_scaled(a,b,c):
    def sig_tmp(x):
        return a / (1 + K.exp(-b*(x-c)))
    return sig_tmp


NNpriceFULL = Sequential() 
NNpriceFULL.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNpriceFULL.add(ZeroPadding2D(padding=(2, 2)))
NNpriceFULL.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='relu'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL.summary()


vega_train1 = np.asarray([vega_train[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
vega_test1 = np.asarray([vega_test[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])
vega_val1 = np.asarray([vega_val[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])

y_train_tmp = np.concatenate((2000*y_train_trafo1_price,2000*vega_train1),axis=1) 
y_test_tmp  = np.concatenate((2000*y_test_trafo1_price,2000*vega_test1),axis=1)
y_val_tmp   = np.concatenate((2000*y_val_trafo1_price,2000*vega_val1),axis=1)
bad_test    = np.min(vega_test, axis=1) <1e-5    
bad_train   = np.min(vega_train, axis=1) <1e-5    
bad_val     = np.min(vega_val, axis=1) <1e-5    
good_test   = np.min(vega_test, axis=1) >= 1e-5    
good_train  = np.min(vega_train, axis=1) >= 1e-5    
good_val    = np.min(vega_val, axis=1) >= 1e-5  
n_valg = np.sum(good_val)
n_testg = np.sum(good_test)
n_traing = np.sum(good_train)
n_valb = Nval-np.sum(good_val)
n_testb = Ntest-np.sum(good_test)
n_trainb = Ntrain-np.sum(good_train)



def ivrmse_approx_no11(y_true_with_vega, y_pred):
    return K.sqrt(81/80*K.mean(K.square((y_pred[:,0,:,:] - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:]))-1/81*K.square((y_pred[:,0,0,0] - y_true_with_vega[:,0,0,0])/y_true_with_vega[:,1,0,0]))
    
def new_norm(y_true_with_vega,y_pred):
    return K.mean(K.norm(K.square((y_pred[:,0,:,:] - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:]),ord="euclidian",dim=(1,2)))
    



#2000 IVRMSE

#NNpriceFULL.compile(loss = ivrmse_approx_no11, optimizer = Adam(clipvalue =1,clipnorm=1))#"adam",metrics=["MAPE",root_mean_squared_error])
#history_LLFullnormal = NNpriceFULL.fit(inputs_train[good_train,:,:,:], y_train_tmp[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], y_val_tmp[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL.save_weights("price_ivrmse11new_weights_1net_2000.h5")
NNpriceFULL.load_weights("price_ivrmse11new_weights_1net_2000.h5")


prediction_iv_test_g   = NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_g    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_iv_test_g,y_test_re_g,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])




meanfullLL_mape = np.mean(err_rel_mat,axis=0)
meanfullLL_mse = np.mean(err_mat,axis=0)
meanfullLL_optll = np.mean(err_optll,axis=0)
meanfullLL_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))
plt.figure()
plt.plot(history_LLFullnormal.history["loss"])
plt.plot(history_LLFullnormal.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()




# 2000 MAPE
#NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = Adam(clipvalue =1,clipnorm=1),metrics=["MAPE","MSE"])#"adam",metrics=["MAPE",root_mean_squared_error])
#history_Fullnormal = NNpriceFULL.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:],2000*y_val_trafo1_price[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL.save_weights("price_rrmse_weights_1net_2000_normal.h5")
NNpriceFULL.load_weights("price_rrmse_weights_1net_2000_normal.h5")
prediction_fullnormal  = NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_g    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_fullnormal,y_test_re_g,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])
plt.figure()
plt.subplot(1,3,1)
plt.plot(history_Fullnormal.history["MAPE"])
plt.plot(history_Fullnormal.history["val_MAPE"])
plt.legend(["MAPE","valMAPE"])
plt.subplot(1,3,2)
plt.plot(history_Fullnormal.history["val_MSE"])
plt.plot(history_Fullnormal.history["MSE"])
plt.legend(["val_MSE","MSE"])
plt.subplot(1,3,3)
plt.plot(history_Fullnormal.history["loss"])
plt.plot(history_Fullnormal.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()
dict_iv ={"price" : y_test_re_g,"forecast" : prediction_fullnormal , "vega": 2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:], "param" : X_test[good_test,:],"rates": rates_test[good_test,:] }
scipy.io.savemat('data_for_IVfullnormal.mat', dict_iv)


prediction_tmp   = NNpriceFULL.predict(np.concatenate((X_tmp_trafo,rates_tmp),axis=1).reshape(459,14,1,1)).reshape((459,9,9))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_tmp,y_price_tmp,data_vega_tmp)
dict_iv_tmp ={"price" : y_price_tmp,"forecast" : prediction_tmp , "vega": data_vega_tmp, "param" : X_tmp,"rates": rates_tmp }
scipy.io.savemat('data_fullnormal.mat',dict_iv_tmp)



meanfull_mape = np.mean(err_rel_mat,axis=0)
meanfull_mse = np.mean(err_mat,axis=0)
meanfull_optll = np.mean(err_optll,axis=0)
meanfull_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))






plt.figure()
plt.subplot(1,2,1)
plt.hist(y_train.flatten(),bins=1000)
plt.hist(y_test.flatten(),bins=1000)
plt.hist(y_val.flatten(),bins=1000)
plt.xlabel("Vola")
plt.legend(["Train","Test","Validation"])
plt.subplot(1,2,2)
plt.hist(y_train_price.flatten(),bins=1000)
plt.hist(y_test_price.flatten(),bins=1000)
plt.hist(y_val_price.flatten(),bins=1000)
plt.xlabel("Price")
plt.legend(["Train","Test","Validation"])
plt.show()






es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 250 ,restore_best_weights=True)
NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = Adam(clipvalue =1,clipnorm=1),metrics=["MAPE","MSE"])#"adam",metrics=["MAPE",root_mean_squared_error])
history_Fullnormal_LONG = NNpriceFULL.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:],2000*y_val_trafo1_price[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL.save_weights("price_rrmse_weights_1net_2000_normal_LONG.h5")
NNpriceFULL.load_weights("price_rrmse_weights_1net_2000_normal_LONG.h5")
prediction_fullnormal_LONG  = NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_g    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_fullnormal,y_test_re_g,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])












# In[Intrinsic Value Penalty:]


NNpriceFULL = Sequential() 
NNpriceFULL.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNpriceFULL.add(ZeroPadding2D(padding=(2, 2)))
NNpriceFULL.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='relu'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL.summary()


intrinsicnet_test  = 2000*np.matlib.repmat(np.matlib.repmat(np.asarray([0.1,0.075,0.05,0.025,0,0,0,0,0]), (9,1,1)).reshape(1,9,9),(n_testg,1,1,1))
intrinsicnet_train  = 2000*np.matlib.repmat(np.matlib.repmat(np.asarray([0.1,0.075,0.05,0.025,0,0,0,0,0]), (9,1,1)).reshape(1,9,9),(n_traing,1,1,1))
intrinsicnet_val  = 2000*np.matlib.repmat(np.matlib.repmat(np.asarray([0.1,0.075,0.05,0.025,0,0,0,0,0]), (9,1,1)).reshape(1,9,9),(n_valg,1,1,1))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = Adam(clipvalue =1,clipnorm=1),metrics=["MAPE","MSE"])#"adam",metrics=["MAPE",root_mean_squared_error])
history_Fullnormal_LONG = NNpriceFULL.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price[good_train,:,:,:]-intrinsicnet_train, batch_size=64, validation_data = (inputs_val[good_val,:,:,:],2000*y_val_trafo1_price[good_val,:,:,:]-intrinsicnet_val), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_rrmse_weights_1net_2000_normal_intrinsic.h5")
NNpriceFULL.load_weights("price_rrmse_weights_1net_2000_normal_intrinsic.h5")
prediction_fullnormal_LONG  = intrinsicnet_test+NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_g    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_fullnormal,y_test_re_g,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])














""""

NNpriceFULL = Sequential() 
NNpriceFULL.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNpriceFULL.add(ZeroPadding2D(padding=(0, 2)))
NNpriceFULL.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(0,2)))
NNpriceFULL.add(Conv2D(64, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(64, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(0,2)))
NNpriceFULL.add(Conv2D(64, (3,2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(0,2)))
NNpriceFULL.add(Conv2D(64, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(64, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(0,2)))
NNpriceFULL.add(Conv2D(64, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULL.add(Conv2D(64, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(64, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(64, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(32, (3, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(2000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='relu'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL.summary()

#2000

NNpriceFULL.compile(loss = ivrmse_approx_no11, optimizer = Adam(clipvalue =2,clipnorm=2))#"adam",metrics=["MAPE",root_mean_squared_error])
NNpriceFULL.fit(inputs_train[good_train,:,:,:], y_train_tmp[good_train,:,:,:], batch_size=128, validation_data = (inputs_val[good_val,:,:,:], y_val_tmp[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_ivrmse11new2_weights_1net_2000.h5")
NNpriceFULL.load_weights("price_ivrmse11new2_weights_1net_2000.h5")


prediction_iv_test_g   = NNpriceFULL.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_g    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_iv_test_g,y_test_re_g,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])

"""



















# In[2.3 CNN as Encoder / Vola Kernel with riskfree rate]:

NNpriceFULLvola = Sequential() 
NNpriceFULLvola.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNpriceFULLvola.add(ZeroPadding2D(padding=(2, 2)))
NNpriceFULLvola.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNpriceFULLvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULLvola.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULLvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULLvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULLvola.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULLvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULLvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULLvola.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULLvola.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULLvola.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULLvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULLvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULLvola.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULLvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULLvola.add(ZeroPadding2D(padding=(2,2)))
NNpriceFULLvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULLvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULLvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceFULLvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceFULLvola.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ="sigmoid"))#, kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULLvola.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULLvola.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='relu'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULLvola.summary()

#NNpriceFULLvola.compile(loss = root_mean_squared_error, optimizer ="adam",metrics=["MAPE","MSE"])
#historyFULLvola = NNpriceFULLvola.fit(inputs_train[good_train,:,:,:], y_train_trafo1[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], y_val_trafo1[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULLvola.save_weights("vola_relmse_weights_1net_full.h5")
NNpriceFULLvola.load_weights("vola_relmse_weights_1net_full.h5")

prediction_vola   = NNpriceFULLvola.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_vola    = y_test_trafo.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat= vola_plotter(prediction_vola,y_test_re_vola)
meanvola_mape = np.mean(err_rel_mat,axis=0)
meanvola_mse = np.mean(err_mat,axis=0)
                       
plt.figure()
plt.subplot(1,3,1)
plt.plot(historyFULLvola.history["MAPE"])
plt.plot(historyFULLvola.history["val_MAPE"])
plt.legend(["MAPE","valMAPE"])
plt.subplot(1,3,2)
plt.plot(historyFULLvola.history["val_MSE"])
plt.plot(historyFULLvola.history["MSE"])
plt.legend(["val_MSE","MSE"])
plt.subplot(1,3,3)
plt.plot(historyFULLvola.history["loss"])
plt.plot(historyFULLvola.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()




from add_func_9x9 import vola_plotter

NNpriceFULL6avola = Sequential() 
NNpriceFULL6avola.add(InputLayer(input_shape=(Nparameters+Nmaturities,1)))
NNpriceFULL6avola.add(ZeroPadding1D(padding=(9)))
NNpriceFULL6avola.add(Conv1D(32, (2), padding='valid',use_bias =True,strides =(1),activation='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(32, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(64, (2),padding='valid',use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(32, (2),padding='valid', use_bias =True,strides =(1),activation ='elu'))
NNpriceFULL6avola.add(Conv1D(9, (2),padding='valid',use_bias =True,strides =(1),activation ="sigmoid"))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL6avola.summary()
#NNpriceFULL6avola.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
#historyFULL6avola = NNpriceFULL6avola.fit(inputs_train[good_train,:,:,0], y_train_trafo1[good_train,0,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,0], y_val_trafo1[good_val,0,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL6avola.save_weights("vola_relmse_weights_1net_1D6a.h5")
NNpriceFULL6avola.load_weights("vola_relmse_weights_1net_1D6a.h5")
prediction_1D6avola   = NNpriceFULL6avola.predict(inputs_test[good_test,:,:,0])
y_test_re_vola    = y_test_trafo.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat= vola_plotter(prediction_1D6avola,y_test_re_vola)
mean6avola_mape = np.mean(err_rel_mat,axis=0)
mean6vola_mse = np.mean(err_mat,axis=0)

plt.figure()
plt.subplot(1,3,1)
plt.plot(historyFULL6avola.history["MAPE"])
plt.plot(historyFULL6avola.history["val_MAPE"])
plt.legend(["MAPE","valMAPE"])
plt.subplot(1,3,2)
plt.plot(historyFULL6avola.history["val_MSE"])
plt.plot(historyFULL6avola.history["MSE"])
plt.legend(["val_MSE","MSE"])
plt.subplot(1,3,3)
plt.plot(historyFULL6avola.history["loss"])
plt.plot(historyFULL6avola.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()







#calibrated parameters
prediction_tmp   = NNpriceFULL6avola.predict(np.concatenate((X_tmp_trafo,rates_tmp),axis=1).reshape(459,14,1))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_tmp,y_vola_tmp,data_vega_tmp)
