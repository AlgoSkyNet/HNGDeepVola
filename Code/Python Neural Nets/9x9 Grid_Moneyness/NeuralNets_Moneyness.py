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
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} ) 
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
K.set_floatx('float64')  

## import variables from the data 
from config_moneyness import Nparameters, Nstrikes, Nmaturities, Ntotal, Ntest, Ntrain, Nval  
from config_moneyness import S0, maturities, moneyness, strikes, strike_net, maturity_net 
from config_moneyness import ub, lb, ub_vola, lb_vola, ub_price, lb_price
from config_moneyness import idx_train,idx_test,idx_val 
from config_moneyness import vega,vega_1
from config_moneyness import rates, rates_net
from config_moneyness import parameters, parameters_trafo   
from config_moneyness import vola, vola_trafo, vola_1, vola_trafo_1       
from config_moneyness import price, price_1, price_trafo_1, intrinsic_net   
## scaling / transformation functions
from config_moneyness import ytransform,yinversetransform,myscale,myinverse
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
        plt.xlim([0,np.max(price)])
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



# In[2. Pricing Network]

## Data for network
param_rates = np.concatenate((parameters_trafo,rates),axis=1).reshape((Ntotal,Nparameters+Nmaturities,1,1))
input_train = param_rates[idx_train,:,:,:]
input_val   = param_rates[idx_val,:,:,:]
input_test  = param_rates[idx_test,:,:,:]
intrinsic_value = (price_1 - intrinsic_net)
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
NNprice_Intrinsic.compile(loss = root_relative_mean_squared_error, optimizer ="adam",metrics=["MAPE","MSE"])
history_intrinsic = NNprice_Intrinsic.fit(input_train, outputs_train, batch_size=32, validation_data = (input_val, outputs_val), epochs =2000, verbose = True, shuffle=0,callbacks=[es])
NNprice_Intrinsic.save_weights("intrinsic_price_rrmse_weights_1net_2000_moneynesss.h5")
#NNprice_Intrinsic.load_weights("intrinsic_price_rrmse_weights_1net_2000_moneynesss.h5")

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
plt.figure()
plt.subplot(1,3,1)
plt.plot(history_price_Intrinsic.history["MAPE"])
plt.plot(history_price_Intrinsic.history["val_MAPE"])
plt.legend(["MAPE","valMAPE"])
plt.subplot(1,3,2)
plt.plot(history_price_Intrinsic.history["val_MSE"])
plt.plot(history_price_Intrinsic.history["MSE"])
plt.legend(["val_MSE","MSE"])
plt.subplot(1,3,3)
plt.plot(history_price_Intrinsic.history["loss"])
plt.plot(history_price_Intrinsic.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()
mean_mape = np.mean(err_rel_mat,axis=0)
mean_mse = np.mean(err_mat,axis=0)
mean_optll = np.mean(err_optll,axis=0)
mean_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))


# DATA FOR IV ANALYSIS
dict_iv ={"price" : y_test_re_g,"forecast" : prediction_fullnormal_LONG , "vega": 2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:], "param" : X_test[good_test,:],"rates": rates_test[good_test,:] }
scipy.io.savemat('data_for_IVfullnormal_intrinsic.mat', dict_iv)



# DATA FOR REAL ANALYSIS:
name_price_tmp = "MLE_calib_price.mat"
name_vola_tmp = "MLE_calib_vola.mat"
name_vega_tmp = "MLE_calib_vega.mat"

path = "D:/GitHub/MasterThesisHNGDeepVola/Code/Python Neural Nets/9x9 Grid/Dataset/"
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
intrinsicnet_tmp = np.zeros((459,9,9))
for i in range(459):
    intrinsicnet_tmp[i,:,:] =   2000*npm.repmat(np.asarray([0.1,0.075,0.05,0.025,0,0,0,0,0]).reshape(1,9), 9,1).reshape(1,9,9)
intrinsicnet_tmp = intrinsicnet_tmp.reshape((459,1,9,9))

prediction_tmp   = (intrinsicnet_tmp+NNpriceFULL.predict(np.concatenate((X_tmp_trafo,rates_tmp),axis=1).reshape(459,14,1,1))).reshape((459,9,9))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp = pricing_plotter(prediction_tmp,y_price_tmp,data_vega_tmp)
dict_iv_tmp ={"price" : y_price_tmp,"forecast" : prediction_tmp , "vega": data_vega_tmp, "param" : X_tmp,"rates": rates_tmp }
scipy.io.savemat('data_fullnormal_intrinsic.mat',dict_iv_tmp)










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
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 250 ,restore_best_weights=True)
NNvola.compile(loss = root_relative_mean_squared_error, optimizer ="adam",metrics=["MAPE","MSE"])
#history_vola = NNvola.fit(input_train, outputs_train, batch_size=64, validation_data = (input_val, outputs_val), epochs =2000, verbose = True, shuffle=1,callbacks=[es])
#NNvola.save_weights("vola_rrmse_weights_1net_moneynesss.h5")
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

#SCALE PRICES BEFORE!!!!!
inputs = price_trafo_1.reshape((Ntotal,Nmaturities,Nstrikes,1))
input_train = inputs[idx_train,:,:,:]
input_val = inputs[idx_val,:,:,:]
input_test = inputs[idx_test,:,:,:]
NNcali = Sequential() 
NNcalibration.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NNcalibration.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='tanh'))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NNcalibration.add(MaxPooling2D(pool_size=(2, 2)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NNcalibration.add(ZeroPadding2D(padding=(1,1)))
NNcalibration.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NNcalibration.add(Flatten())
NNcalibration.add(Dense(Nparameters,activation = 'linear',use_bias=True,kernel_constraint = tf.keras.constraints.NonNeg()))
NNcalibration.summary()
#setting
NNcalibration.compile(loss =log_constraint(param=1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
es = EarlyStopping(monitor='val_MSE', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
history_calibration = NNcalibration.fit(input_train,train_trafo2, batch_size=120, validation_data = (y_val_price_scale,X_val_trafo2), epochs=100, verbose = True, shuffle=1,callbacks =[es])
NNcalibration.save_weights("calibrationweights_price_scale.h5")
NNcalibration.load_weights("calibrationweights_price_scale.h5")

prediction_calibration1 = NNcalibration.predict(y_test_price_scale)
prediction_invtrafo1= np.array([myinverse(x) for x in prediction_calibration1])

#plots
error_cal1,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp = calibration_plotter(prediction_calibration1,X_test_trafo2,X_test)




NNcalibration.compile(loss =log_constraint(param=0.01,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
#es = EarlyStopping(monitor='val_MAPE', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
#history = NNcalibration.fit(y_train_price_scale,X_train_trafo2, batch_size=120, validation_data = (y_val_price_scale,X_val_trafo2), epochs=100, verbose = True, shuffle=1,callbacks =[es])
#NNcalibration.save_weights("calibrationweights_price_scale2.h5")
NNcalibration.load_weights("calibrationweights_price_scale.h5")


prediction_calibration2 = NNcalibration.predict(y_test_price_scale)
prediction_invtrafo2= np.array([myinverse(x) for x in prediction_calibration2])

#plots
error_cal2,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp = calibration_plotter(prediction_calibration2,X_test_trafo2,X_test)










