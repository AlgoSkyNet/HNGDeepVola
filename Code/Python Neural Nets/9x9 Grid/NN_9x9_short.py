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
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} ) 
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


dict_iv ={"price" : y_test_re_g,"forecast" : prediction_fullnormal , "vega": 2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:], "param" : X_test[good_test,:],"rates": rates_test[good_test,:] }
scipy.io.savemat('data_for_IVfullnormal.mat', dict_iv)











# In[Intrinsic Value Penalty:]

from tensorflow.compat.v1.keras.optimizers import Adam
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
NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(1000,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceFULL.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='relu'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceFULL.summary()
import numpy.matlib as npm
strike_net = 2000*npm.repmat(np.asarray([0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1]).reshape(1,9), 9,1)
maturity_net = 1/252*npm.repmat(np.asarray([10,40,70,100,130,160,190,220,250]).reshape(9,1), 1,9)
intrinsicnet_test = [];
for i in range(n_testg):
    rates_net = npm.repmat(rates_test[good_test,:][i,:].reshape((9,1)),1,9)
    tmp = 2000-np.exp(-rates_net*maturity_net)*strike_net
    tmp[tmp<0] = 0
    intrinsicnet_test.append(tmp)
intrinsicnet_test = np.asarray(intrinsicnet_test).reshape(n_testg,1,9,9)
intrinsicnet_train = [];
for i in range(n_traing):
    rates_net = npm.repmat(rates_train[good_train,:][i,:].reshape((9,1)),1,9)
    tmp = 2000-np.exp(-rates_net*maturity_net)*strike_net
    tmp[tmp<0] = 0
    intrinsicnet_train.append(tmp)
intrinsicnet_train = np.asarray(intrinsicnet_train).reshape(n_traing,1,9,9)
intrinsicnet_val = [];
for i in range(n_valg):
    rates_net = npm.repmat(rates_val[good_val,:][i,:].reshape((9,1)),1,9)
    tmp = 2000-np.exp(-rates_net*maturity_net)*strike_net
    tmp[tmp<0] = 0
    intrinsicnet_val.append(tmp)
intrinsicnet_val = np.asarray(intrinsicnet_val).reshape(n_valg,1,9,9)
pos_ratio1 = np.mean(2000*y_train_trafo1_price[good_train,:,:,:]>intrinsicnet_train)
pos_ratio2 = np.mean(2000*y_test_trafo1_price[good_test,:,:,:]>intrinsicnet_test)
pos_ratio3 = np.mean(2000*y_val_trafo1_price[good_val,:,:,:]>intrinsicnet_val)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = Adam(clipvalue =1,clipnorm=1),metrics=["MAPE","MSE"])#"adam",metrics=["MAPE",root_mean_squared_error])
history_Fullnormal_LONG = NNpriceFULL.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price[good_train,:,:,:]-intrinsicnet_train, batch_size=64, validation_data = (inputs_val[good_val,:,:,:],2000*y_val_trafo1_price[good_val,:,:,:]-intrinsicnet_val), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
NNpriceFULL.save_weights("price_rrmse_weights_1net_2000_normal_intrinsic2.h5")
#NNpriceFULL.load_weights("price_rrmse_weights_1net_2000_normal_intrinsic2.h5")
prediction_fullnormal_LONG  = (intrinsicnet_test+NNpriceFULL.predict(inputs_test[good_test,:,:,:])).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_g    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_fullnormal_LONG,y_test_re_g,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])
plt.figure()
plt.subplot(1,3,1)
plt.yscale("log")
plt.plot(history_Fullnormal_LONG.history["val_MAPE"])
plt.plot(history_Fullnormal_LONG.history["MAPE"])

plt.legend(["MAPE","valMAPE"])
plt.subplot(1,3,2)
plt.yscale("log")
plt.plot(history_Fullnormal_LONG.history["val_MSE"])
plt.plot(history_Fullnormal_LONG.history["MSE"])

plt.legend(["val_MSE","MSE"])
plt.subplot(1,3,3)
plt.yscale("log")
plt.plot(history_Fullnormal_LONG.history["val_loss"])
plt.plot(history_Fullnormal_LONG.history["loss"])
plt.legend(["loss","val_loss"])
plt.show()
meanfull_mape = np.mean(err_rel_mat,axis=0)
meanfull_mse = np.mean(err_mat,axis=0)
meanfull_optll = np.mean(err_optll,axis=0)
meanfull_ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))


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
