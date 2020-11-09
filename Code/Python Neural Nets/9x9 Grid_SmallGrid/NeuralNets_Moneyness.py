# In[0. Preambel]:

import numpy as np
import numpy.matlib as npm
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.compat.v1.keras.layers import Reshape,InputLayer,Dense,Flatten, Conv2D,Conv1D, Dropout, Input,ZeroPadding2D,ZeroPadding1D,MaxPooling2D
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 88} ) 
#sess = tf.compat.v1.Session(config=config) 
#K.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
K.set_floatx('float64')  

## import variables from the data 
from config_moneyness2 import Nparameters, Nstrikes, Nmaturities, Ntotal, Ntest, Ntrain, Nval  
from config_moneyness2 import S0, maturities, moneyness, strikes, strike_net, maturity_net 
from config_moneyness2 import ub, lb, ub_vola, lb_vola, ub_price, lb_price
from config_moneyness2 import idx_train,idx_test,idx_val 
from config_moneyness2 import vega,vega_1
from config_moneyness2 import rates, rates_net
from config_moneyness2 import parameters, parameters_trafo   
from config_moneyness2 import vola, vola_trafo, vola_1, vola_trafo_1       
from config_moneyness2 import price, price_1, price_trafo_1, intrinsic_net   
## scaling / transformation functions
from config_moneyness2 import ytransform,yinversetransform,myscale,myinverse
##custom errors
from add_func_moneyness import root_mean_squared_error,root_relative_mean_squared_error,mse_constraint,rmse_constraint
from add_func_moneyness import constraint_violation,log_constraint,miss_count,mape_constraint
from add_func_moneyness import log_constraint,miss_count,mape_constraint

## plotting
from add_func_moneyness import pricing_plotter,plotter_autoencoder,calibration_plotter,vola_plotter

def sig_scaled(a,b,c,d):
    def sig_tmp(x):
        return a / (1 + K.exp(-b*(x-c)))+d
    return sig_tmp



# In[1. Data Summary]:

## Descriptive Statistics
mean_price = np.mean(price_1,axis=0)
std_price  = np.std(price_1,axis=0)
min_price  = np.min(price_1,axis=0)
max_price  = np.max(price_1,axis=0)
mean_vola  = np.mean(vola_1,axis=0)
std_vola   = np.std(vola_1,axis=0)
min_vola   = np.min(vola_1,axis=0)
max_vola   = np.max(vola_1,axis=0)
mean_vega  = np.mean(vega_1,axis=0)
std_vega   = np.std(vega_1,axis=0)
min_vega   = np.min(vega_1,axis=0)
max_vega   = np.max(vega_1,axis=0)

fig = plt.figure()
plt.suptitle("Distribution of True Prices per Gridpoint")
for i in range(Nmaturities):
    for j in range(Nstrikes):
        plt.subplot(Nmaturities,Nstrikes,Nmaturities*i+j+1)
        plt.hist(price_1[:,i,j].flatten(),bins=100)
        #plt.xlim([0,np.max(price)])
for ax in fig.get_axes():
    ax.label_outer()
plt.show()

fig = plt.figure()
plt.suptitle("Distribution of True Volas per Gridpoint")
for i in range(Nmaturities):
    for j in range(Nstrikes):
        plt.subplot(Nmaturities,Nstrikes,Nmaturities*i+j+1)
        plt.hist(vola_1[:,i,j].flatten(),bins=100)
        plt.xlim([0,np.max(vola)])
for ax in fig.get_axes():
    ax.label_outer()
plt.show()
summary_paramters= np.asarray([np.min(parameters,axis=0),np.quantile(parameters,0.05,axis=0),np.quantile(parameters,0.25,axis=0),np.median(parameters,axis=0),np.mean(parameters,axis=0),np.quantile(parameters,0.75,axis=0),np.quantile(parameters,0.95,axis=0),np.max(parameters,axis=0)])
corre_parameters = np.corrcoef(parameters.T)

fig = plt.figure()
plt.hist(price_1[:,0,8].flatten(),bins=100)
# In[2. Pricing Network]

## Data for network
param_rates = np.concatenate((parameters_trafo,rates),axis=1).reshape((Ntotal,Nparameters+Nmaturities,1,1))
input_train = param_rates[idx_train,:,:,:]
input_val   = param_rates[idx_val,:,:,:]
input_test  = param_rates[idx_test,:,:,:]
intrinsic_value = (price_1 - intrinsic_net)
intrinsic_mean = np.mean(intrinsic_value,axis=0)
intrinsic_std = np.std(intrinsic_value,axis=0)
ub_i        = np.amax(intrinsic_value)
lb_i         = np.amin(intrinsic_value)
def intrinsicScale(y,scaler):
    if scaler==1: #-1,1
        return (y - (ub_i + lb_i)*0.5) * 2 / (ub_i - lb_i)
    elif scaler==2 :#0,1
        return 0.5*((y - (ub_i + lb_i)*0.5) * 2 / (ub_i - lb_i) +1)
    elif scaler==3: #1,2
        return 0.5*((y - (ub_i + lb_i)*0.5) * 2 / (ub_i - lb_i) +1)+1
    else:
        return y
def intrinsicInverse(y,scaler=1):
    if scaler==1: #-1,1
        return (2*y-1)*(ub_i - lb_i) *0.5 + (ub_i + lb_i)*0.5
    elif scaler ==2: #0,1
        return (2*y-1)*(ub_i - lb_i) *0.5 + (ub_i + lb_i)*0.5
    elif scaler ==3: #1,2
        return (2*(y-1)-1)*(ub_i - lb_i) *0.5 + (ub_i + lb_i)*0.5
    else:
        return y
    
def mape_scale(y_true,y_pred):
    return 100*K.mean(K.abs((intrinsicInverse(y_true,0)-intrinsicInverse(y_pred,0))/intrinsicInverse(y_true,0)))
outputs        = intrinsicScale(intrinsic_value,0).reshape((Ntotal,1,9,9))
outputs_train  = outputs[idx_train,:,:,:]
outputs_val    = outputs[idx_val,:,:,:]
outputs_test   = outputs[idx_test,:,:,:]

## Architecture
NNprice_Intrinsic = Sequential() 
NNprice_Intrinsic.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_Intrinsic.add(ZeroPadding2D(padding=(2, 2)))
NNprice_Intrinsic.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNprice_Intrinsic.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_Intrinsic.add(ZeroPadding2D(padding=(2,2)))
NNprice_Intrinsic.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_Intrinsic.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_Intrinsic.add(ZeroPadding2D(padding=(2,2)))
NNprice_Intrinsic.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_Intrinsic.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_Intrinsic.add(ZeroPadding2D(padding=(2,2)))
NNprice_Intrinsic.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_Intrinsic.add(ZeroPadding2D(padding=(2,2)))
NNprice_Intrinsic.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_Intrinsic.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_Intrinsic.add(ZeroPadding2D(padding=(2,2)))
NNprice_Intrinsic.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_Intrinsic.add(ZeroPadding2D(padding=(2,2)))
NNprice_Intrinsic.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_Intrinsic.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_Intrinsic.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_Intrinsic.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_Intrinsic.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(S0/2,1,0,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNprice_Intrinsic.summary()

## Training
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 150 ,restore_best_weights=True)
NNprice_Intrinsic.compile(loss = root_relative_mean_squared_error, optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=["MAPE","MSE"])
history_intrinsic = NNprice_Intrinsic.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =2000, verbose = True, shuffle=0,callbacks=[es])
NNprice_Intrinsic.compile(loss = root_relative_mean_squared_error, optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=["MAPE","MSE"])
history_intrinsic = NNprice_Intrinsic.fit(input_train, outputs_train, batch_size=120, validation_data = (input_val, outputs_val), epochs =2000, verbose = True, shuffle=0,callbacks=[es])
#NNprice_Intrinsic.save_weights("intrinsic_price_rrmse_weights_1net_2000_moneynesss.h5")
#NNprice_Intrinsic.load_weights("intrinsic_price_rrmse_weights_1net_2000_moneynesss.h5")
#NNprice_Intrinsic.compile(loss = root_relative_mean_squared_error, optimizer =Adam(clipvalue=10,learning_rate=5e-6),metrics=["MAPE","MSE"])
#history_intrinsic = NNprice_Intrinsic.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =2000, verbose = True, shuffle=0,callbacks=[es])
#NNprice_Intrinsic.save_weights("intrinsic_price_rrmse_weights_1net_2000_moneynesss.h5")

## Training
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#NNprice_Intrinsic.compile(loss = "MSE", optimizer ="adam",metrics=["MAPE"])
#history_intrinsic = NNprice_Intrinsic.fit(input_train, outputs_train, batch_size=64, validation_data = (input_val, outputs_val), epochs =2000, verbose = True, shuffle=0,callbacks=[es])
#NNprice_Intrinsic.save_weights("intrinsic_price_mse_weights_1net_2000_moneynesss.h5")
#NNprice_Intrinsic.load_weights("intrinsic_price_mse_weights_1net_2000_moneynesss.h5")

# Results
prediction_intrinsic  = intrinsic_net[idx_test,:,:]+intrinsicInverse(NNprice_Intrinsic.predict(input_test).reshape((Ntest,Nmaturities,Nstrikes)),0)
price_test           = price_1[idx_test,:,:]
vega_test            = vega_1[idx_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_intrinsic,price_test,vega_test)
idx_high = np.zeros((Ntest,1))
for i in range(Ntest):
    idx_high[i] = np.all(price_test[i,:,:]>0.05)
idx_high = np.reshape(idx_high==1,(Ntest,))
err_rel_mat_high = 100*err_rel_mat[idx_high,:,:]
mean_high = np.mean(err_rel_mat_high,axis=0)
pricing_plotter(prediction_intrinsic[idx_high,:,:],price_test[idx_high,:,:],vega_test[idx_high,:,:])

plt.figure()
plt.subplot(1,3,1)
plt.plot(history_intrinsic.history["MAPE"])
plt.plot(history_intrinsic.history["val_MAPE"])
plt.legend(["MAPE","valMAPE"])
plt.subplot(1,3,2)
plt.plot(history_intrinsic.history["val_MSE"])
plt.plot(history_intrinsic.history["MSE"])
plt.legend(["val_MSE","MSE"])
plt.subplot(1,3,3)
plt.plot(history_intrinsic.history["loss"])
plt.plot(history_intrinsic.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()
mean_mape = np.mean(err_rel_mat,axis=0)
mean_mse = np.mean(err_mat,axis=0)
mean_optll = np.mean(err_optll,axis=0)
mean_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))


# DATA FOR IV ANALYSIS
dict_iv ={"price" : price_test,"forecast" : prediction_intrinsic , "vega":vega_1[idx_test,:,:], "param" : parameters[idx_test,:],"rates": rates[idx_test,:] }
scipy.io.savemat('data_for_IV_moneyness.mat', dict_iv)

# DATA FOR REAL ANALYSIS:
name_price_tmp = "MLE_calib_price_Moneyness.mat"
name_vola_tmp = "MLE_calib_vola_Moneyness.mat"
name_vega_tmp = "MLE_calib_vega_Moneyness.mat"

path = "C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Python Neural Nets/9x9 Grid_Moneyness/"
tmp         = scipy.io.loadmat(path+name_vola_tmp)
data_vola_tmp        =tmp['data_vola']
tmp         = scipy.io.loadmat(path+name_price_tmp)
data_price_tmp       = tmp['data_price']
tmp         = scipy.io.loadmat(path+name_vega_tmp)
data_vega_tmp       = tmp['data_vega'].reshape((129,9,9))



X_tmp = data_vola_tmp[:,:5]
X_tmp_trafo = np.array([myscale(x) for x in X_tmp])
rates_tmp2 = data_vola_tmp[:,5:14]
y_vola_tmp = data_vola_tmp[:,14:].reshape((129,9,9))
y_price_tmp = data_price_tmp[:,14:].reshape((129,9,9))
intrinsicnet_tmp = np.zeros((129,9,9))
intrinsic_net_realdata  = []
rates_net_realdata      = []

for i in range(129):
    rates_tmp  = npm.repmat(rates_tmp2[i,:].reshape((9,1)),1,9)
    tmp        = S0-np.exp(-rates_tmp*maturity_net)*strike_net
    tmp[tmp<0] = 0
    rates_net_realdata.append(rates_tmp)
    intrinsic_net_realdata.append(tmp)
intrinsic_net_realdata     = np.asarray(intrinsic_net_realdata).reshape(129,9,9)
rates_net_realdata          = np.asarray(rates_net_realdata).reshape(129,9,9)

prediction_tmp   = intrinsic_net_realdata+NNprice_Intrinsic.predict(np.concatenate((X_tmp_trafo,rates_tmp2),axis=1).reshape((129,14,1,1))).reshape((129,9,9))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_tmp,y_price_tmp,data_vega_tmp)
mean_mape_realdata = np.mean(err_rel_mat,axis=0)
mean_mse_realdata  = np.mean(err_mat,axis=0)

dict_iv_tmp ={"price" : y_price_tmp,"forecast" : prediction_tmp , "vega": data_vega_tmp, "param" : X_tmp,"rates": rates_tmp }
scipy.io.savemat('data_fullnormal_intrinsic.mat',dict_iv_tmp)

# DATA FOR REAL ANALYSIS FULL:
name_price_tmp = "MLE_calib_price_full.mat"
name_vola_tmp = "MLE_calib_vola_full.mat"
name_vega_tmp = "MLE_calib_vega_full.mat"

path = "C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Python Neural Nets/9x9 Grid_Moneyness/"
tmp         = scipy.io.loadmat(path+name_vola_tmp)
data_vola_tmp        =tmp['data_vola']
tmp         = scipy.io.loadmat(path+name_price_tmp)
data_price_tmp       = tmp['data_price']
tmp         = scipy.io.loadmat(path+name_vega_tmp)
data_vega_tmp       = tmp['data_vega'].reshape((458,9,9))



X_tmp = data_vola_tmp[:,:5]
X_tmp_trafo = np.array([myscale(x) for x in X_tmp])
rates_tmp2 = data_vola_tmp[:,5:14]
y_vola_tmp = data_vola_tmp[:,14:].reshape((458,9,9))
y_price_tmp = data_price_tmp[:,14:].reshape((458,9,9))
intrinsicnet_tmp = np.zeros((458,9,9))
intrinsic_net_realdata  = []
rates_net_realdata      = []

for i in range(458):
    rates_tmp  = npm.repmat(rates_tmp2[i,:].reshape((9,1)),1,9)
    tmp        = S0-np.exp(-rates_tmp*maturity_net)*strike_net
    tmp[tmp<0] = 0
    rates_net_realdata.append(rates_tmp)
    intrinsic_net_realdata.append(tmp)
intrinsic_net_realdata     = np.asarray(intrinsic_net_realdata).reshape(458,9,9)
rates_net_realdata          = np.asarray(rates_net_realdata).reshape(458,9,9)

prediction_tmp   = intrinsic_net_realdata+NNprice_Intrinsic.predict(np.concatenate((X_tmp_trafo,rates_tmp2),axis=1).reshape((458,14,1,1))).reshape((458,9,9))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_tmp,y_price_tmp,data_vega_tmp)
mean_mape_realdata = np.mean(err_rel_mat,axis=0)
mean_mse_realdata  = np.mean(err_mat,axis=0)

dict_iv_tmp ={"price" : y_price_tmp,"forecast" : prediction_tmp , "vega": data_vega_tmp, "param" : X_tmp,"rates": rates_tmp2 }
scipy.io.savemat('data_intrinsic_smallgird.mat',dict_iv_tmp)










# In[3. Vola Network]:

## Data for Network
param_rates = np.concatenate((parameters_trafo,rates),axis=1).reshape((Ntotal,Nparameters+Nmaturities,1,1))
input_train = param_rates[idx_train,:,:,:]
input_val   = param_rates[idx_val,:,:,:]
input_test  = param_rates[idx_test,:,:,:]
outputs     = (vola_1).reshape((Ntotal,1,9,9))
outputs_train  = outputs[idx_train,:,:,:]
outputs_val    = outputs[idx_val,:,:,:]
outputs_test   = outputs[idx_test,:,:,:]

## Architecture
NNvola = Sequential() 
NNvola.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1)))
NNvola.add(ZeroPadding2D(padding=(2, 2)))
NNvola.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNvola.add(ZeroPadding2D(padding=(2,2)))
NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNvola.add(ZeroPadding2D(padding=(2,2)))
NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNvola.add(ZeroPadding2D(padding=(2,2)))
NNvola.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNvola.add(ZeroPadding2D(padding=(2,2)))
NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNvola.add(ZeroPadding2D(padding=(2,2)))
NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNvola.add(ZeroPadding2D(padding=(2,2)))
NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNvola.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ="sigmoid"))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNvola.summary()

## Training
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 150 ,restore_best_weights=True)
NNvola.compile(loss = root_relative_mean_squared_error, optimizer =Adam(learning_rate=1e-4),metrics=["MAPE","MSE"])
history_vola = NNvola.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =2000, verbose = True, shuffle=1,callbacks=[es])
NNvola.save_weights("vola_rrmse_weights_1net_moneynesss.h5")
NNvola.load_weights("vola_rrmse_weights_1net_moneynesss.h5")



# CUSTOM EARLY STOPPING based on average historic loss

MAX_EPOCH = 1000
EARLY_PATIENCE = 50
MEAN_PATIENCE = 10
history_total = []
loss_train = []
loss_val = []
mean_valloss = []
patience = 0
best_idx = 0
best_meanloss = 1000000000000
for n in range(MAX_EPOCH):
    print([n,patience,best_idx])
    tmp_hist = NNvola.fit(input_train, outputs_train, batch_size=64, validation_data = (input_val, outputs_val), epochs =1, verbose = True, shuffle=1)
    history_total.append(tmp_hist)
    loss_train.append(tmp_hist.history["loss"])
    loss_val.append(tmp_hist.history["val_loss"])
    tmp_loss = np.mean(loss_val[np.max([n-MEAN_PATIENCE+1,0]):n+1])
    mean_valloss.append(tmp_loss)   
    if best_meanloss>tmp_loss:
        best_meanloss=tmp_loss
        NNvola.save_weights("best_weights.h5")
        best_idx = n
        patience = 0
    else:
        patience +=1
        if patience >=EARLY_PATIENCE:
            break            
NNvola.load_weights("best_weights.h5")
def network_build():
    NNvola = Sequential() 
    NNvola.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1)))
    NNvola.add(ZeroPadding2D(padding=(2, 2)))
    NNvola.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
    NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
    NNvola.add(ZeroPadding2D(padding=(2,2)))
    NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
    NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
    NNvola.add(ZeroPadding2D(padding=(2,2)))
    NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
    NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
    NNvola.add(ZeroPadding2D(padding=(2,2)))
    NNvola.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
    NNvola.add(ZeroPadding2D(padding=(2,2)))
    NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
    NNvola.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
    NNvola.add(ZeroPadding2D(padding=(2,2)))
    NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
    NNvola.add(ZeroPadding2D(padding=(2,2)))
    NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
    NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
    NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
    NNvola.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
    NNvola.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ="sigmoid"))#, kernel_constraint = tf.keras.constraints.NonNeg()))
    NNvola.summary()
    NNvola.compile(loss = root_relative_mean_squared_error, optimizer ="adam",metrics=["MAPE","MSE"])

    return NNvola

#as function module:
def networkFit(nn_name,input_train, outputs_train,input_val, outputs_val,MAX_EPOCH = 1000,EARLY_PATIENCE = 50,MEAN_PATIENCE = 10,BATCH_SIZE = 64):
    Network= network_build() 
    history_total = []
    loss_train = []
    loss_val = []
    mean_valloss = []
    patience = 0
    best_idx = 0
    best_meanloss = 1000000000000
    for n in range(MAX_EPOCH):
        print([n,patience,best_idx])
        tmp_hist = Network.fit(input_train, outputs_train, batch_size=BATCH_SIZE, validation_data = (input_val, outputs_val), epochs =1, verbose = True, shuffle=1)
        history_total.append(tmp_hist)
        loss_train.append(tmp_hist.history["loss"])
        loss_val.append(tmp_hist.history["val_loss"])
        tmp_loss = np.mean(loss_val[np.max([n-MEAN_PATIENCE+1,0]):n+1])
        mean_valloss.append(tmp_loss)   
        if best_meanloss>tmp_loss:
            best_meanloss=tmp_loss
            Network.save_weights("best_weights.h5")
            best_idx = n
            patience = 0
        else:
            patience +=1
            if patience >=EARLY_PATIENCE:
                break            
            Network.load_weights("best_weights.h5")
    return history_total,loss_train,loss_val,mean_valloss,best_meanloss,best_idx
history_total,loss_train,loss_val,mean_valloss,best_meanloss,best_idx  = networkFit("NNvola",input_train, outputs_train,input_val, outputs_val)
 




 



# Results
prediction_vola   = NNvola.predict(input_test).reshape((Ntest,Nmaturities,Nstrikes))
vola_test = vola_1[idx_test,:,:]
err_rel_mat,err_mat= vola_plotter(prediction_vola,vola_test)
meanvola_mape = np.mean(err_rel_mat,axis=0)
meanvola_mse = np.mean(err_mat,axis=0)
# DATA FOR IV ANALYSIS
dict_iv ={"vola" : vola_test,"vola_forecast" : prediction_vola, "vega":vega_1[idx_test,:,:], "param" : parameters[idx_test,:],"rates": rates[idx_test,:] }
scipy.io.savemat('data_for_IVvola_moneyness.mat', dict_iv)
                 
plt.figure()
plt.subplot(1,3,1)
plt.plot(history_vola.history["MAPE"])
plt.plot(history_vola.history["val_MAPE"])
plt.legend(["MAPE","valMAPE"])
plt.subplot(1,3,2)
plt.plot(history_vola.history["val_MSE"])
plt.plot(history_vola.history["MSE"])
plt.legend(["val_MSE","MSE"])
plt.subplot(1,3,3)
plt.plot(history_vola.history["loss"])
plt.plot(history_vola.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()



# In[Calibration]:
"""
#SCALE PRICES BEFORE!!!!!
inputs =np.concatenate((price_trafo_1.reshape((Ntotal,Nmaturities,Nstrikes,1)),rates_net.reshape((Ntotal,Nmaturities,Nstrikes,1))),axis=3)

input_train = inputs[idx_train,:,:,:]
input_val = inputs[idx_val,:,:,:]
input_test = inputs[idx_test,:,:,:]
NNcalibration = Sequential() 
NNcalibration.add(InputLayer(input_shape=(Nmaturities,Nstrikes,2)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(MaxPooling2D(pool_size=(2, 2)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Flatten())
NNcalibration.add(Dense(Nparameters,activation = sig_scaled(2,1,0,-1),use_bias=True))
NNcalibration.summary()
es = EarlyStopping(monitor='val_MSE', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
NNcalibration.compile(loss =log_constraint(param=1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib1 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=240, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])
NNcalibration.compile(loss =log_constraint(param=0.1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib2 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=120, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])
NNcalibration.compile(loss =log_constraint(param=0.05,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib3 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=120, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])#
NNcalibration.compile(loss =log_constraint(param=0.02,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib4 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=120, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.save_weights("calibrationweights_price_scale.h5")
NNcalibration.load_weights("calibrationweights_price_scale.h5")
#error mean in %: [ 12.65046436   1.88101337  18.22682322 121.62693572  16.03671059]
#error median in %: [ 3.37519672  1.35066321  8.92520557 10.98832693  1.12920175]
#error max in % : [3.72135249e+04, 3.01093580e+02, 5.76727687e+04, 7.32935607e+05, 3.59270113e+05]
#error 95qu in % :[ 53.01078992,   4.80586492,  32.07408447, 144.67171103,       6.03768085]
#error 5qu in % : [0.2385793 , 0.13190749, 0.77972109, 0.95170923, 0.10152413]
#errror 75qu in % : [12.33816926,  2.25984515, 15.42927149, 24.77151527,  2.1740383 ]
#errror 25qu in % : [1.25614122, 0.65717499, 4.1210808 , 4.8363425 , 0.5123959 ]

prediction_calibration1 = NNcalibration.predict(input_test)
prediction_invtrafo1= np.array([myinverse(x) for x in prediction_calibration1])
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios = calibration_plotter(prediction_calibration1,parameters_trafo[idx_test,:],parameters[idx_test,:])











#SCALE PRICES BEFORE!!!!!
inputs =np.concatenate((price_trafo_1.reshape((Ntotal,Nmaturities,Nstrikes,1)),vola_1.reshape((Ntotal,Nmaturities,Nstrikes,1)),rates_net.reshape((Ntotal,Nmaturities,Nstrikes,1))),axis=3)

input_train = inputs[idx_train,:,:,:]
input_val = inputs[idx_val,:,:,:]
input_test = inputs[idx_test,:,:,:]
NNcalibration = Sequential() 
NNcalibration.add(InputLayer(input_shape=(Nmaturities,Nstrikes,3)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(MaxPooling2D(pool_size=(2, 2)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Flatten())
NNcalibration.add(Dense(Nparameters,activation = sig_scaled(2,1,0,-1),use_bias=True))
NNcalibration.summary()
es = EarlyStopping(monitor='val_MSE', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
NNcalibration.compile(loss =log_constraint(param=1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib1 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=240, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])
#0.0070
NNcalibration.compile(loss =log_constraint(param=0.1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib2 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=120, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])
#0.0019
NNcalibration.compile(loss =log_constraint(param=0.05,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib3 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=120, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])#
NNcalibration.compile(loss =log_constraint(param=0.02,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
es = EarlyStopping(monitor='val_MSE', mode='min', verbose=1,patience = 100 ,restore_best_weights=True)
history_calib4 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=120, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.save_weights("calibrationweights_pricevola.h5")

prediction_calibration1 = NNcalibration.predict(input_test)
prediction_invtrafo1= np.array([myinverse(x) for x in prediction_calibration1])
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios = calibration_plotter(prediction_calibration1,parameters_trafo[idx_test,:],parameters[idx_test,:])
summary_calibration = 100*np.asarray([np.quantile(error,0.05,axis=0),np.quantile(error,0.25,axis=0),np.median(error,axis=0),np.mean(error,axis=0),np.quantile(error,0.75,axis=0),np.quantile(error,0.95,axis=0),np.max(error,axis=0)])
#STACKING WITH PENALISATION?



#SCALE PRICES BEFORE!!!!!
inputs =np.concatenate((vola_1.reshape((Ntotal,Nmaturities,Nstrikes,1)),rates_net.reshape((Ntotal,Nmaturities,Nstrikes,1))),axis=3)

input_train = inputs[idx_train,:,:,:]
input_val = inputs[idx_val,:,:,:]
input_test = inputs[idx_test,:,:,:]
NNcalibration = Sequential() 
NNcalibration.add(InputLayer(input_shape=(Nmaturities,Nstrikes,2)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(MaxPooling2D(pool_size=(2, 2)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Flatten())
NNcalibration.add(Dense(Nparameters,activation = sig_scaled(2,1,0,-1),use_bias=True))
NNcalibration.summary()
es = EarlyStopping(monitor='val_MSE', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
NNcalibration.compile(loss =log_constraint(param=1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib1 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=240, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])
#0.0070
NNcalibration.compile(loss =log_constraint(param=0.1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib2 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=120, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])
#0.0019
NNcalibration.compile(loss =log_constraint(param=0.05,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib3 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=120, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])#
NNcalibration.compile(loss =log_constraint(param=0.02,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib4 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=120, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=1000, verbose = True, shuffle=1,callbacks =[es])
NNcalibration.save_weights("calibrationweights_vola.h5")

prediction_calibration1 = NNcalibration.predict(input_test)
prediction_invtrafo1= np.array([myinverse(x) for x in prediction_calibration1])
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios = calibration_plotter(prediction_calibration1,parameters_trafo[idx_test,:],parameters[idx_test,:])
summary_calibration = 100*np.asarray([np.quantile(error,0.05,axis=0),np.quantile(error,0.25,axis=0),np.median(error,axis=0),np.mean(error,axis=0),np.quantile(error,0.75,axis=0),np.quantile(error,0.95,axis=0),np.max(error,axis=0)])
#STACKING WITH PENALISATION?
"""



####

#SCALE PRICES BEFORE!!!!!
inputs =np.concatenate((price_trafo_1.reshape((Ntotal,Nmaturities,Nstrikes,1)),vola_1.reshape((Ntotal,Nmaturities,Nstrikes,1)),rates_net.reshape((Ntotal,Nmaturities,Nstrikes,1))),axis=3)

input_train = inputs[idx_train,:,:,:]
input_val = inputs[idx_val,:,:,:]
input_test = inputs[idx_test,:,:,:]
NNcalibration = Sequential() 
NNcalibration.add(InputLayer(input_shape=(Nmaturities,Nstrikes,3)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibration.add(Flatten())
NNcalibration.add(Dense(Nparameters,activation = sig_scaled(2,1,0,-1),use_bias=True))
NNcalibration.summary()
#es = EarlyStopping(monitor='val_MSE', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#NNcalibration.compile(loss =log_constraint(param=1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
#history_calib1 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=250, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.compile(loss =log_constraint(param=0.1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
#history_calib2 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=250, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.compile(loss =log_constraint(param=0.02,p2=15), optimizer = Adam(learning_rate=1e-4),metrics=["MAPE", "MSE",miss_count])
#history_calib3 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=250, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.compile(loss =log_constraint(param=0.005,p2=15), optimizer = Adam(learning_rate=1e-4),metrics=["MAPE", "MSE",miss_count])
#history_calib4 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=250, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.compile(loss =log_constraint(param=0.0005,p2=15), optimizer = Adam(learning_rate=1e-4),metrics=["MAPE", "MSE",miss_count])
#history_calib5 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=250, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.compile(loss =log_constraint(param=0.0002,p2=15), optimizer = Adam(learning_rate=5e-5),metrics=["MAPE", "MSE",miss_count])
#history_calib6 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=250, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.compile(loss =log_constraint(param=0.00015,p2=15), optimizer = Adam(learning_rate=5e-6),metrics=["MAPE", "MSE",miss_count])
#history_calib7 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=250, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.compile(loss =log_constraint(param=0.00005,p2=15), optimizer = Adam(learning_rate=2e-6),metrics=["MAPE", "MSE",miss_count])
#history_calib8 = NNcalibration.fit(input_train,parameters_trafo[idx_train,:], batch_size=250, validation_data = (input_val,parameters_trafo[idx_val,:]), epochs=2000, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.save_weights("calibrationweights_full2.h5")
#error mean in %: [ 1.56521672  0.18926635  2.26517553 19.22380365  6.11809311  0.23928737]
#error median in %: [0.31152869 0.10567252 0.35819373 0.78040178 0.19235505 0.14658808]
#1.5e-5
NNcalibration.load_weights("calibrationweights_full2.h5")


prediction_calibration1 = NNcalibration.predict(input_test)
prediction_invtrafo1= np.array([myinverse(x) for x in prediction_calibration1])
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios = calibration_plotter(prediction_invtrafo1,parameters[idx_test,:])
summary_calibration = 100*np.asarray([np.quantile(error,0.05,axis=0),np.quantile(error,0.25,axis=0),np.median(error,axis=0),np.mean(error,axis=0),np.quantile(error,0.75,axis=0),np.quantile(error,0.95,axis=0),np.max(error,axis=0)])

#STACKING WITH PENALISATION?

def log_constraint_mix(param,p2=30):
    def log_mse_constraint_mix(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*(ub[0] - lb[0])+(ub[0] + lb[0]))
            traf_g = 0.5*(y_pred[:,2]*(ub[2] - lb[2])+(ub[2] + lb[2]))
            traf_b = 0.5*(y_pred[:,1]*(ub[1] - lb[1])+(ub[1] + lb[1]))
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.mean(K.square(y_pred - y_true))*(1+param*K.mean(1/(1+K.exp(-p2*(constraint-1)))))
    return log_mse_constraint_mix
def log_constraint2(param,p2=30):
    def log_mse_constraint2(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*(ub[0] - lb[0])+(ub[0] + lb[0]))
            traf_g = 0.5*(y_pred[:,2]*(ub[2] - lb[2])+(ub[2] + lb[2]))
            traf_b = 0.5*(y_pred[:,1]*(ub[1] - lb[1])+(ub[1] + lb[1]))
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.mean(K.square(y_pred - y_true)) +param*K.mean(1/(1+K.exp(-p2*(-(constraint-0.2))))+1/(1+K.exp(-p2*(constraint-1.1))))
    return log_mse_constraint2
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios = calibration_plotter(prediction_invtrafo1[:237,:],parameters[idx_test,:][:237,:])
summary_calibration = 100*np.asarray([np.quantile(error,0.05,axis=0),np.quantile(error,0.25,axis=0),np.median(error,axis=0),np.mean(error,axis=0),np.quantile(error,0.75,axis=0),np.quantile(error,0.95,axis=0),np.max(error,axis=0)])
dict_calib ={"price_calibtrue":price_1[idx_test,:,:], "params_calib" : prediction_invtrafo1, "vega_calib":vega_1[idx_test,:,:], "param_true" : parameters[idx_test,:],"rates_calib": rates[idx_test,:] }
scipy.io.savemat('data_calib_moneyness.mat', dict_calib)

# In[Surface to Surface Autoencoder]
input_test = np.concatenate((prediction_calibration1,rates[idx_test,:]),axis=1).reshape((Ntest,Nparameters+Nmaturities,1,1))
forecast = intrinsic_net[idx_test,:,:]+intrinsicInverse(NNprice_Intrinsic.predict(input_test).reshape((Ntest,Nmaturities,Nstrikes)),0)

mape,mse = plotter_autoencoder(forecast,price_1[idx_test,:,:],testing_violation,testing_violation2)


# In[Parameter to Parameter Autoencoder]
input_test  = param_rates[idx_test,:,:,:]
prediction_intrinsic  = intrinsic_net[idx_test,:,:]+intrinsicInverse(NNprice_Intrinsic.predict(input_test).reshape((Ntest,Nmaturities,Nstrikes)),0)
vola_test = vola_1[idx_test,:,:]
inputs_auto =np.concatenate((prediction_intrinsic.reshape((Ntest,Nmaturities,Nstrikes,1)),vola_test.reshape((Ntest,Nmaturities,Nstrikes,1)),rates_net[idx_test,:].reshape((Ntest,Nmaturities,Nstrikes,1))),axis=3)

forecast_param = NNcalibration.predict(inputs_auto)
prediction_invtrafo1= np.array([myinverse(x) for x in forecast_param])
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios = calibration_plotter(forecast_param,parameters[idx_test,:])





# In[MissingValueNetwork]:

n1 = 30000
n2 = 500
"""
mv_x_train_set = np.zeros((n1*n2,9,9,3))
mv_y_train_set = np.zeros((n1*n2,9,9,1))
#mv_x_train_set = []
#mv_y_train_set = []
idx = 0
basic_0 = np.zeros((9,9,1))
price_reshape = price_1.reshape((Ntotal,9,9,1))
for i in range(n1):
    if i%500==0:
        print(i)
    tmp2 = price_reshape[i,:,:,:]
    for j in range(n2):
        count_mv = np.random.randint(10,50)
        tmp = np.concatenate((tmp2,basic_0,rates_net[i,:,:].reshape(9,9,1)),axis=2)
        for k in range(count_mv):
            pos1 = np.random.randint(0,9)
            pos2 = np.random.randint(0,9)
            tmp[pos1,pos2,0] = -999
            tmp[pos1,pos2,1] = 1    
        mv_x_train_set[idx,:,:,:] = tmp
        mv_y_train_set[idx,:,:,:] = tmp2
        idx +=1
"""
np.load('/mv_x_train.npy')
np.load('/mv_y_train.npy')

Im2Im  = Sequential()       
Im2Im.add(InputLayer(input_shape=(9,9,3,)))
Im2Im.add(ZeroPadding2D(padding=(1, 1)))
Im2Im.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
Im2Im.add(Conv2D(64, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
Im2Im.add(ZeroPadding2D(padding=(1,1)))
Im2Im.add(Conv2D(128, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
Im2Im.add(Conv2D(64, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
Im2Im.add(ZeroPadding2D(padding=(1,1)))
Im2Im.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
Im2Im.add(Conv2D(1, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
Im2Im.summary()
es_im2im = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 5 ,restore_best_weights=True)
Im2Im.compile(loss = "MSE", optimizer = Adam(learning_rate=1e-3),metrics=["MAPE"])
Im2Im_history = Im2Im.fit(mv_x_train_set[:int(0.8*n1*n2),:,:,:], mv_y_train_set[:int(0.8*n1*n2),:,:,:], batch_size=2500, validation_data = (mv_x_train_set[int(0.8*n1*n2):int(0.9*n1*n2),:,:,:], mv_y_train_set[int(0.8*n1*n2):int(0.9*n1*n2),:,:,:]),epochs =50, verbose = True,callbacks =[es_im2im], use_multiprocessing=True,workers=50)
#Im2Im.save_weights("missing_value_network_mse.h5")
Im2Im.load_weights("missing_value_network_mse.h5")
#Im2Im.compile(loss = root_relative_mean_squared_error, optimizer = Adam(learning_rate=1e-2),metrics=["MAPE","MSE"])
#Im2Im_history = Im2Im.fit(mv_x_train_set[:int(0.8*n1*n2),:,:,:], mv_y_train_set[:int(0.8*n1*n2),:,:,:], batch_size=5000, validation_data = (mv_x_train_set[int(0.8*n1*n2):int(0.9*n1*n2),:,:,:], mv_y_train_set[int(0.8*n1*n2):int(0.9*n1*n2),:,:,:]),epochs =50, verbose = True,callbacks =[es_im2im], use_multiprocessing=True,workers=50)
#Im2Im.save_weights("missing_value_network_relmse.h5")
prediction_Im2Im  = Im2Im.predict(mv_x_train_set[int(0.9*n1*n2):,:,:,:]).reshape((int(n1*n2*0.1),Nmaturities,Nstrikes))

vola_plotter(prediction_Im2Im,mv_y_train_set[int(0.9*n1*n2):,:,:,:].reshape((int(n1*n2*0.1),Nmaturities,Nstrikes)))

