# In[Preambel]:
import numpy as np
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.compat.v1.keras.layers import Reshape,InputLayer,Dense,Flatten, Conv2D,Conv1D, Dropout, Input,ZeroPadding2D,ZeroPadding1D,MaxPooling2D
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import EarlyStopping, TerminateOnNaN

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

# In[Additional Graphs]:
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


# In[Calibration Network without interest rates]:
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
from add_func_9x9 import log_constraint,miss_count,mape_constraint,l2rel_log_constraint
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
NN2s.load_weights("calibrationweights_price_scale2.h5")


prediction_calibration2 = NN2s.predict(y_test_price_scale)
prediction_invtrafo2= np.array([myinverse(x) for x in prediction_calibration2])

#plots
error_cal2,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp = calibration_plotter(prediction_calibration2,X_test_trafo2,X_test)

def sig_scaled2(a,b,c,d):
    def sig_tmp(x):
        return a / (1 + K.exp(-b*(x-c)))-d
    return sig_tmp

NN2t = Sequential() 
NN2t.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2t.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2t.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2t.add(MaxPooling2D(pool_size=(2, 2)))
NN2t.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2t.add(ZeroPadding2D(padding=(1,1)))
NN2t.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2t.add(ZeroPadding2D(padding=(1,1)))
NN2t.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2t.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2t.add(ZeroPadding2D(padding=(1,1)))
NN2t.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2t.add(ZeroPadding2D(padding=(1,1)))
NN2t.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2t.add(Flatten())
NN2t.add(Dense(Nparameters,activation = sig_scaled2(2,1,0,1),use_bias=True))
NN2t.summary()
#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
from add_func_9x9 import log_constraint,miss_count,mape_constraint
#setting
#NN2.compile(loss =mse_constraint(0.75), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
NN2t.compile(loss =log_constraint(param=0.02,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
history_calib = NN2t.fit(y_train_price_scale,X_train_trafo2, batch_size=120, validation_data = (y_val_price_scale,X_val_trafo2), epochs=1000, verbose = True, shuffle=1,callbacks =[es,TerminateOnNaN()])
#NN2t.load_weights("calibrationweights_price_scale2.h5") #trained with 0.05
#error mean in %: [ 11.49460257   3.31107766  47.6081939  269.84902779   2.02618329]
#error median in %: [ 3.91680533  2.1041695  21.11251931 28.47064688  1.54899471]

NN2t.load_weights("calibrationweights_price_scale3.h5")
prediction_calibration2 = NN2t.predict(y_test_price_scale)
prediction_invtrafo2= np.array([myinverse(x) for x in prediction_calibration2])

#plots
error_cal2,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,bad_scenarios = calibration_plotter(prediction_calibration2,X_test_trafo2,X_test)
#error mean in %: [  9.59446253   3.3221475   61.35534678 257.13928187   1.98673457]
#error median in %: [ 3.9278435   2.2255678  13.32730602 24.6635999   1.34492209]

prediction_calibration_train = NN2t.predict(y_train_price_scale)
error_cal_train,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,bad_scenarios_train = calibration_plotter(prediction_calibration_train,X_train_trafo2,X_train)
#violation error mean in %: [6.74096068 7.22387983 5.4740058  6.37823168 0.87656606]
#no violation error mean in %: [ 14.63622309   3.23196808  91.11594699 295.82874936   2.95922869]
#violation error median in %: [6.74096068 7.22387983 5.4740058  6.37823168 0.87656606]
#no violation error median in %: [ 3.85857103  2.20160393 13.15030205 23.89956761  1.33736569]
#error mean in %: [ 14.63607029   3.23204534  91.11428951 295.82314745   2.95918838]
#error median in %: [ 3.85865632  2.2017053  13.15001254 23.89923041  1.33733563]




# decrease param1 every 200 epochs (20patience) with param =  1 ,0.1, 0.05, 0.02 
# 0.02 leads to violations which fit really good luckily
#hyperparameter league; optimised grid search?,
path = "D:/GitHub/MasterThesisHNGDeepVola/Code/Python Neural Nets/9x9 Grid/Dataset/"
mat         = scipy.io.loadmat(path+"2010_interpolatedgrid_full.mat")
data_calib1        = ytransform(mat['data_1']/2000,0)
mat         = scipy.io.loadmat(path+"2010_interpolatedgrid_mv.mat")
data_calib2       = ytransform(mat['data_2']/2000,0)
calib_mv1 = np.array([myinverse(x) for x in NN2t.predict(data_calib1.reshape(50,9,9,1))])
calib_mv2 = np.array([myinverse(x) for x in NN2t.predict(Im2Im.predict(data_calib2))])


NN2u = Sequential() 
NN2u.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2u.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2u.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2u.add(MaxPooling2D(pool_size=(2, 2)))
NN2u.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2u.add(ZeroPadding2D(padding=(1,1)))
NN2u.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2u.add(ZeroPadding2D(padding=(1,1)))
NN2u.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2u.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2u.add(ZeroPadding2D(padding=(1,1)))
NN2u.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2u.add(ZeroPadding2D(padding=(1,1)))
NN2u.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled2(2,1,0,1)))
NN2u.add(Flatten())
NN2u.add(Dense(Nparameters,activation = sig_scaled2(2,1,0,1),use_bias=True))
NN2u.summary()
es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
#NN2u.compile(loss =log_constraint(param=1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
#history_calib1 = NN2u.fit(y_train_trafo2_price,X_train_trafo2, batch_size=120, validation_data = (y_val_trafo2_price,X_val_trafo2), epochs=1000, verbose = True, shuffle=1,callbacks =[es,TerminateOnNaN()])
#NN2u.compile(loss =log_constraint(param=0.1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
#history_calib2 = NN2u.fit(y_train_trafo2_price,X_train_trafo2, batch_size=120, validation_data = (y_val_trafo2_price,X_val_trafo2), epochs=1000, verbose = True, shuffle=1,callbacks =[es,TerminateOnNaN()])
#NN2u.compile(loss =log_constraint(param=0.05,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
#history_calib3 = NN2u.fit(y_train_trafo2_price,X_train_trafo2, batch_size=120, validation_data = (y_val_trafo2_price,X_val_trafo2), epochs=1000, verbose = True, shuffle=1,callbacks =[es,TerminateOnNaN()])
#NN2u.compile(loss =log_constraint(param=0.02,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
#history_calib3 = NN2u.fit(y_train_trafo2_price,X_train_trafo2, batch_size=120, validation_data = (y_val_trafo2_price,X_val_trafo2), epochs=1000, verbose = True, shuffle=1,callbacks =[es,TerminateOnNaN()])
NN2u.load_weights("calibrationweights_price_noscale.h5")
prediction_calibration2= NN2u.predict(y_test_trafo2_price)
prediction_invtrafo2= np.array([myinverse(x) for x in prediction_calibration2])
#plots
error_cal2,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,bad_scenarios = calibration_plotter(prediction_calibration2,X_test_trafo2,X_test)
#violation error mean in %: [8.41421396 9.89426299 5.57732473 9.79142987 0.41425912]
#no violation error mean in %: [ 10.84411513   3.83723956  33.30644314 270.48082549   1.95264569]
#violation error median in %: [8.41421396 9.89426299 5.57732473 9.79142987 0.41425912]
#no violation error median in %: [ 4.53147174  2.48029833 12.27446847 28.76238562  1.39845012]
#error mean in %: [ 10.84400188   3.83752187  33.30515071 270.46867497   1.95257399]
#error median in %: [ 4.53164383  2.48037491 12.27347941 28.76128185  1.39836023]
#NN2u.compile(loss =log_constraint(param=0.02,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
#history_calib3 = NN2u.fit(y_train_trafo2_price,X_train_trafo2, batch_size=120, validation_data = (y_val_trafo2_price,X_val_trafo2), epochs=1000, verbose = True, shuffle=1,callbacks =[es,TerminateOnNaN()])
es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=1,patience = 30 ,restore_best_weights=True)

NN2u.compile(loss =log_constraint(param=0.01,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib4 = NN2u.fit(y_train_trafo2_price,X_train_trafo2, batch_size=120, validation_data = (y_val_trafo2_price,X_val_trafo2), epochs=1000, verbose = True, shuffle=1,callbacks =[es,TerminateOnNaN()])

prediction_calibration2= NN2u.predict(y_test_trafo2_price)
prediction_invtrafo2= np.array([myinverse(x) for x in prediction_calibration2])
#plots
error_cal2,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,bad_scenarios = calibration_plotter(prediction_calibration2,X_test_trafo2,X_test)
#NN2u.save_weights("calibrationweights_price_noscale2.h5")
NN2u.load_weights("calibrationweights_price_noscale2.h5")

#violation error mean in %: [ 8.43303848 19.71777368  3.59140641 23.68943221  2.49093993]
#no violation error mean in %: [  8.34937103   2.46414277  46.06680183 246.88935609   1.80947501]
#violation error median in %: [ 7.71004472 13.32453395  3.37973814 11.49818568  1.9625375 ]
#no violation error median in %: [ 3.64995966  1.05854563 10.57111727 28.80930643  1.16830942]
#error mean in %: [  8.34938273   2.46655531  46.0608626  246.8581466    1.8095703 ]
#error median in %: [ 3.65276139  1.05874232 10.56845929 28.80609638  1.16847505]

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



#  In[Pricing Network / Pricing Kernel with riskfree rate]:
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











# In[Intrinsic Value Penalty MAPE]

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

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#NNpriceFULL.compile(loss = root_relative_mean_squared_error, optimizer = Adam(clipvalue =1,clipnorm=1),metrics=["MAPE","MSE"])#"adam",metrics=["MAPE",root_mean_squared_error])
#history_Fullnormal_LONG = NNpriceFULL.fit(inputs_train[good_train,:,:,:], 2000*y_train_trafo1_price[good_train,:,:,:]-intrinsicnet_train, batch_size=64, validation_data = (inputs_val[good_val,:,:,:],2000*y_val_trafo1_price[good_val,:,:,:]-intrinsicnet_val), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL.save_weights("price_rrmse_weights_1net_2000_normal_intrinsic2.h5")
NNpriceFULL.load_weights("price_rrmse_weights_1net_2000_normal_intrinsic2.h5")
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

# In[Intrinsic Value Option Likelyhood]
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

def ivrmse_approx(y_true_with_vega, y_pred):
    return K.sqrt(K.mean(K.square((y_pred - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:])))

def mape_intr(y_true_with_vega, y_pred):
    return 100*(K.mean(K.abs((y_pred - y_true_with_vega[:,0,:,:])/(y_true_with_vega[:,0,:,:]+y_true_with_vega[:,2,:,:]))))
def mse_intr(y_true_with_vega, y_pred):
    return K.mean(K.square((y_pred - y_true_with_vega[:,0,:,:])))
def option_log_likelyhood(y_true_with_vega, y_pred):
        return K.mean(K.log(K.square((y_pred - y_true_with_vega[:,0,:,:])/y_true_with_vega[:,1,:,:])))

y_intr_train = np.concatenate((2000*y_train_trafo1_price[good_train,:,:,:]-intrinsicnet_train,2000*vega_train1[good_train,:,:,:],intrinsicnet_train),axis=1) 
y_intr_test  = np.concatenate((2000*y_test_trafo1_price[good_test,:,:,:]-intrinsicnet_test,2000*vega_test1[good_test,:,:,:],intrinsicnet_test),axis=1)
y_intr_val   = np.concatenate((2000*y_val_trafo1_price[good_val,:,:,:]-intrinsicnet_val,2000*vega_val1[good_val,:,:,:],intrinsicnet_val),axis=1)

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#NNpriceFULL.compile(loss = ivrmse_approx, optimizer = Adam(clipvalue =1,clipnorm=1),metrics=[mape_intr,mse_intr])#"adam",metrics=["MAPE",root_mean_squared_error])
#history_Fullnormal_LONG_optll = NNpriceFULL.fit(inputs_train[good_train,:,:,:], y_intr_train, batch_size=64, validation_data = (inputs_val[good_val,:,:,:],y_intr_val), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULL.save_weights("price_rrmse_weights_1net_2000_normal_intrinsic_optll.h5")
NNpriceFULL.load_weights("price_rrmse_weights_1net_2000_normal_intrinsic_optll.h5")

prediction_fullnormal_iv  = (intrinsicnet_test+NNpriceFULL.predict(inputs_test[good_test,:,:,:])).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_g    = 2000*yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_fullnormal_iv,y_test_re_g,2000*vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])

plt.figure()
plt.subplot(1,3,1)
plt.yscale("log")
plt.plot(history_Fullnormal_LONG_optll.history["val_mape_intr"])
plt.plot(history_Fullnormal_LONG_optll.history["mape_intr"])

plt.legend(["MAPE","valMAPE"])
plt.subplot(1,3,2)
plt.yscale("log")
plt.plot(history_Fullnormal_LONG_optll.history["val_mse_intr"])
plt.plot(history_Fullnormal_LONG_optll.history["mse_intr"])

plt.legend(["val_MSE","MSE"])
plt.subplot(1,3,3)
plt.yscale("log")
plt.plot(history_Fullnormal_LONG_optll.history["val_loss"])
plt.plot(history_Fullnormal_LONG_optll.history["loss"])
plt.legend(["loss","val_loss"])
plt.show()
meanfull_optll_mape = np.mean(err_rel_mat,axis=0)
meanfull_optll__mse = np.mean(err_mat,axis=0)
meanfull_optll__optll = np.mean(err_optll,axis=0)
meanfull_optll__ivrmse = np.sqrt(np.mean(err_iv_approx,axis=0))



# In[Volatilitly Network / Vola Kernel with riskfree rate]:

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
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#NNpriceFULLvola.compile(loss = root_mean_squared_error, optimizer ="adam",metrics=["MAPE","MSE"])
#historyFULLvola = NNpriceFULLvola.fit(inputs_train[good_train,:,:,:], y_train_trafo1[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], y_val_trafo1[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULLvola.save_weights("vola_relmse_weights_1net_full50pat.h5")
NNpriceFULLvola.load_weights("vola_relmse_weights_1net_full50pat.h5")

#NNpriceFULLvola.load_weights("vola_relmse_weights_1net_full.h5")
from add_func_9x9 import vola_plotter

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






#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#NNpriceFULLvola.compile(loss = "MSE", optimizer ="adam",metrics=["MAPE"])
#historyFULLvola_mse = NNpriceFULLvola.fit(inputs_train[good_train,:,:,:], y_train_trafo1[good_train,:,:,:], batch_size=64, validation_data = (inputs_val[good_val,:,:,:], y_val_trafo1[good_val,:,:,:]), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceFULLvola.save_weights("vola_mse_weights_1net_full.h5")
NNpriceFULLvola.load_weights("vola_mse_weights_1net_full.h5")




prediction_vola_mse   = NNpriceFULLvola.predict(inputs_test[good_test,:,:,:]).reshape((n_testg,Nmaturities,Nstrikes))
y_test_re_vola    = y_test_trafo.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat= vola_plotter(prediction_vola,y_test_re_vola)
meanvola2_mape = np.mean(err_rel_mat,axis=0)
meanvola2_mse = np.mean(err_mat,axis=0)
                       
plt.figure()
plt.subplot(1,2,1)
plt.plot(historyFULLvola_mse.history["mean_absolute_percentage_error"])
plt.plot(historyFULLvola_mse.history["val_mean_absolute_percentage_error"])
plt.legend(["MAPE","valMAPE"])
plt.subplot(1,2,2)
plt.plot(historyFULLvola_mse.history["loss"])
plt.plot(historyFULLvola_mse.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()


# In[Simultaniuos Training]:
NNpriceSimul = Sequential() 
NNpriceSimul.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNpriceSimul.add(ZeroPadding2D(padding=(2, 2)))
NNpriceSimul.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNpriceSimul.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceSimul.add(ZeroPadding2D(padding=(2,2)))
NNpriceSimul.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceSimul.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceSimul.add(ZeroPadding2D(padding=(2,2)))
NNpriceSimul.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceSimul.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceSimul.add(ZeroPadding2D(padding=(2,2)))
NNpriceSimul.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceSimul.add(ZeroPadding2D(padding=(2,2)))
NNpriceSimul.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceSimul.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNpriceSimul.add(ZeroPadding2D(padding=(2,2)))
NNpriceSimul.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceSimul.add(ZeroPadding2D(padding=(2,2)))
NNpriceSimul.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceSimul.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceSimul.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceSimul.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNpriceSimul.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation =sig_scaled(1,1,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceSimul.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='sigmoid', kernel_constraint = tf.keras.constraints.NonNeg()))
#NNpriceSimul.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='relu'))#, kernel_constraint = tf.keras.constraints.NonNeg()))
NNpriceSimul.summary()
import numpy.matlib as npm
strike_net = npm.repmat(np.asarray([0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1]).reshape(1,9), 9,1)
maturity_net = 1/252*npm.repmat(np.asarray([10,40,70,100,130,160,190,220,250]).reshape(9,1), 1,9)
intrinsicnet_test = [];
for i in range(n_testg):
    rates_net = npm.repmat(rates_test[good_test,:][i,:].reshape((9,1)),1,9)
    tmp = 1-np.exp(-rates_net*maturity_net)*strike_net
    tmp[tmp<0] = 0
    intrinsicnet_test.append(tmp)
intrinsicnet_test = np.asarray(intrinsicnet_test).reshape(n_testg,1,9,9)
intrinsicnet_train = [];
for i in range(n_traing):
    rates_net = npm.repmat(rates_train[good_train,:][i,:].reshape((9,1)),1,9)
    tmp = 1-np.exp(-rates_net*maturity_net)*strike_net
    tmp[tmp<0] = 0
    intrinsicnet_train.append(tmp)
intrinsicnet_train = np.asarray(intrinsicnet_train).reshape(n_traing,1,9,9)
intrinsicnet_val = [];
for i in range(n_valg):
    rates_net = npm.repmat(rates_val[good_val,:][i,:].reshape((9,1)),1,9)
    tmp = 1-np.exp(-rates_net*maturity_net)*strike_net
    tmp[tmp<0] = 0
    intrinsicnet_val.append(tmp)
intrinsicnet_val = np.asarray(intrinsicnet_val).reshape(n_valg,1,9,9)
pos_ratio1 = np.mean(y_train_trafo1_price[good_train,:,:,:]>intrinsicnet_train)
pos_ratio2 = np.mean(y_test_trafo1_price[good_test,:,:,:]>intrinsicnet_test)
pos_ratio3 = np.mean(y_val_trafo1_price[good_val,:,:,:]>intrinsicnet_val)

train_simul = np.concatenate((y_train_trafo1_price[good_train,:,:,:]-intrinsicnet_train,y_train_trafo1[good_train,:,:,:]),axis= 1)
test_simul  = np.concatenate((y_test_trafo1_price[good_test,:,:,:]-intrinsicnet_test,y_test_trafo1[good_test,:,:,:]),axis= 1)
val_simul   = np.concatenate((y_val_trafo1_price[good_val,:,:,:]-intrinsicnet_val,y_val_trafo1[good_val,:,:,:]),axis= 1)

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#NNpriceSimul.compile(loss = root_relative_mean_squared_error, optimizer = Adam(clipvalue =1),metrics=["MAPE","MSE"])#"adam",metrics=["MAPE",root_mean_squared_error])
#history_Simulnormal_LONG = NNpriceSimul.fit(inputs_train[good_train,:,:,:],train_simul , batch_size=200, validation_data = (inputs_val[good_val,:,:,:],val_simul), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#NNpriceSimul.save_weights("price_rrmse_weights_1net_simul.h5")
NNpriceSimul.load_weights("price_rrmse_weights_1net_simul.h5")
test_add = np.concatenate((intrinsicnet_test,np.zeros_like(y_test_trafo1[good_test,:,:,:])),axis= 1)
prediction_Simul  = test_add+NNpriceSimul.predict(inputs_test[good_test,:,:,:])
prediction_sim_vola = prediction_Simul[:,1,:,:].reshape((n_testg,9,9))
prediction_sim_price = prediction_Simul[:,0,:,:].reshape((n_testg,9,9))

mape_simul = np.mean(np.abs((prediction_Simul-test_simul)/test_simul)) 
mse_simul = np.mean(np.square(prediction_Simul-test_simul)) 
y_test_re_vola    = y_test_trafo.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
y_test_re_g    = yinversetransform(y_test_trafo_price).reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_sim_price,y_test_re_g,vega_test.reshape((Ntest,Nmaturities,Nstrikes))[good_test,:,:])
err_rel_mat,err_mat= vola_plotter(prediction_sim_vola,y_test_re_vola)
plt.figure()
plt.subplot(1,3,1)
plt.yscale("log")
plt.plot(history_Simulnormal_LONG.history["mean_absolute_percentage_error"])
plt.plot(history_Simulnormal_LONG.history["val_mean_absolute_percentage_error"])
plt.legend(["MAPE","valMAPE"])
plt.subplot(1,3,2)
plt.yscale("log")
plt.plot(history_Simulnormal_LONG.history["mean_squared_error"])
plt.plot(history_Simulnormal_LONG.history["val_mean_squared_error"])
plt.legend(["val_MSE","MSE"])
plt.subplot(1,3,3)
plt.yscale("log")
plt.plot(history_Simulnormal_LONG.history["loss"])
plt.plot(history_Simulnormal_LONG.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()

#
mean_price_test = np.mean(2000*y_test_re_g,axis=0)
mean_vola_test  = np.mean(y_test_re_vola,axis=0)
std_price_test  = np.std(2000*y_test_re_g,axis=0)
std_vola_test   = np.std(y_test_re_vola,axis=0)


# In[Missing Value Network]
# The idea is to train a network which fills missing values
# for every surface create n surfaces with a random amount of values at random positions replaced by -999
# fit the missing value surfaces to the true surfaces

n1 = 30000
n2 = 500
mv_x_train_set = np.zeros((n1*n2,9,9,2))
mv_y_train_set = np.zeros((n1*n2,9,9,1))
#mv_x_train_set = []
#mv_y_train_set = []
idx = 0
basic_0 = np.zeros((9,9,1))
for i in range(n1):
    tmp2 = 2000*y_train_trafo2_price[i,:,:,:]
    for j in range(n2):
        count_mv = np.random.randint(10,45)
        tmp = np.concatenate((tmp2,basic_0),axis=2)
        for k in range(count_mv):
            pos1 = np.random.randint(0,9)
            pos2 = np.random.randint(0,9)
            tmp[pos1,pos2,0] = -999
            tmp[pos1,pos2,1] = 1
        mv_x_train_set[idx,:,:,:] = tmp
        mv_y_train_set[idx,:,:,:] = tmp2
        idx +=1
        #mv_x_train_set.append(tmp)
        #mv_y_train_set.append(2000*y_train_trafo2_price[i,:,:,:])
#mv_x_train_set = np.asarray(mv_x_train_set).reshape((n1*n2,9,9,2))
#mv_y_train_set = np.asarray(mv_y_train_set).reshape((n1*n2,9,9,1))
#dict_mv ={"X" : mv_x_train_set, "Y": mv_y_train_set }
#scipy.io.savemat('data_Missing_Value.mat',dict_mv)

Im2Im = Sequential() 
Im2Im.add(InputLayer(input_shape=(9,9,2,)))
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
#Im2Im.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE","MSE"])
#Im2Im_history = Im2Im.fit(mv_x_train_set[:int(0.8*n1*n2),:,:,:], mv_y_train_set[:int(0.8*n1*n2),:,:,:], batch_size=2500, validation_data = (mv_x_train_set[int(0.8*n1*n2):int(0.9*n1*n2),:,:,:], mv_y_train_set[int(0.8*n1*n2):int(0.9*n1*n2),:,:,:]),epochs =50, verbose = True,callbacks =[es_im2im], use_multiprocessing=True,workers=50)
#Im2Im.load_weights("missing_value_network_mse.h5")
Im2Im.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
Im2Im_history = Im2Im.fit(mv_x_train_set[:int(0.8*n1*n2),:,:,:], mv_y_train_set[:int(0.8*n1*n2),:,:,:], batch_size=2500, validation_data = (mv_x_train_set[int(0.8*n1*n2):int(0.9*n1*n2),:,:,:], mv_y_train_set[int(0.8*n1*n2):int(0.9*n1*n2),:,:,:]),epochs =50, verbose = True,callbacks =[es_im2im], use_multiprocessing=True,workers=50)
Im2Im.save_weights("missing_value_network_relmse.h5")


prediction_im2im   = Im2Im.predict(mv_x_train_set[int(0.9*n1*n2):,:,:,:]).reshape((int(0.1*n1*n2),9,9))
err_rel_mat,err_mat = vola_plotter(prediction_im2im,mv_y_train_set[int(0.9*n1*n2):,:,:,:].reshape((int(0.1*n1*n2),9,9)))
Im2Im_history_mape = np.mean(err_rel_mat,axis=0)
Im2Im_history_mse = np.mean(err_mat,axis=0)
missing_value_error = mv_x_train_set[int(0.9*n1*n2):,:,:,1]*err_rel_mat;
missing_value_error2 = mv_x_train_set[int(0.9*n1*n2):,:,:,1]*err_mat;
mape_Im2Im_mv =np.zeros((9,9))
mse_Im2Im_mv =np.zeros((9,9))
counter_mv = np.sum(mv_x_train_set[int(0.9*n1*n2):,:,:,1],axis=0)
for i in range(9):
    for j in range(9):
        tmp = missing_value_error[:,i,j];
        tmp = tmp[tmp>0]
        mape_Im2Im_mv[i,j] = np.mean(tmp)
        tmp = missing_value_error2[:,i,j];
        tmp = tmp[tmp>0]
        mse_Im2Im_mv[i,j] = np.mean(tmp) 
plt.figure()
plt.subplot(1,2,1)
#plt.yscale("log")
plt.plot(Im2Im_history.history["MAPE"])
plt.plot(Im2Im_history.history["val_MAPE"])
plt.legend(["MAPE","valMAPE"])
plt.subplot(1,2,2)
#plt.yscale("log")
plt.plot(Im2Im_history.history["MSE"])
plt.plot(Im2Im_history.history["val_MSE"])
plt.legend(["val_MSE","MSE"])
plt.show()
