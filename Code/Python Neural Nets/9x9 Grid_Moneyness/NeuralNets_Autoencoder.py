# In[0. Preambel]:

import numpy as np
import numpy.matlib as npm
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.compat.v1.keras.layers import Concatenate,Reshape,InputLayer,Dense,Flatten, Conv2D,Conv1D, Dropout, Input,ZeroPadding2D,ZeroPadding1D,MaxPooling2D
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
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


# In[AutencoderNetwork]:
intrinsic_value = (price_1 - intrinsic_net)
input_1 = price_1.reshape((Ntotal,9,9,1))
input_2 = rates
output_1 = intrinsic_value.reshape((Ntotal,1,9,9))
output_2 = parameters_trafo
outputs_train = [output_1[idx_train,:],output_2[idx_train,:]]
outputs_val   = [output_1[idx_val,:],output_2[idx_val,:]]
outputs_test  = [output_1[idx_test,:],output_2[idx_test,:]]
input_train   = [input_1[idx_train,:],input_2[idx_train,:]]
input_val     = [input_1[idx_val,:],input_2[idx_val,:]]
input_test    = [input_1[idx_test,:],input_2[idx_test,:]]

InputCalib = Input(shape=(Nmaturities,Nstrikes,1))
calib = ZeroPadding2D(padding=(1,1))(InputCalib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = ZeroPadding2D(padding=(1,1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = ZeroPadding2D(padding=(1,1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = ZeroPadding2D(padding=(1,1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib)
calib = Flatten()(calib)
calib = Dense(Nparameters,activation = sig_scaled(2,1,0,-1),use_bias=True)(calib)
calib = Model(inputs = InputCalib,outputs= calib)
#calib.load_weights("calibrationweights_Price.h5")

InputInterest = Input(shape=(Nmaturities,))
interest = Model(inputs=InputInterest,outputs= InputInterest)

combinate = Concatenate(axis=1)([calib.output,interest.output])

pricer = Reshape((Nparameters+Nmaturities,1,1))(combinate)
pricer = ZeroPadding2D(padding=(2,2))(pricer)
pricer = Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu')(pricer)
pricer = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer)
pricer = ZeroPadding2D(padding=(2,2))(pricer)
pricer = Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu')(pricer)
pricer = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer)
pricer = ZeroPadding2D(padding=(2,2))(pricer)
pricer = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer)
pricer = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer)
pricer = ZeroPadding2D(padding=(2,2))(pricer)
pricer = Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer)
pricer = ZeroPadding2D(padding=(2,2))(pricer)
pricer = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer)
pricer = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer)
pricer = ZeroPadding2D(padding=(2,2))(pricer)
pricer = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu')(pricer)
pricer = ZeroPadding2D(padding=(2,2))(pricer)
pricer = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu')(pricer)
pricer = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu')(pricer)
pricer = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer)
pricer = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu')(pricer)
pricer = Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(S0/2,1,0,0))(pricer)
Autoencoder = Model(inputs =[calib.input,interest.input],outputs=[pricer,calib.output])

def autoencoder_loss(weight_losses=0.5,weight_constraints=1,param=15):
    def weighted_loss(y_true,y_pred):
        l1 = K.sqrt(K.mean(K.square((y_pred[0] - y_true[0])/y_true[0])))
        traf_a = 0.5*(y_pred[1][:,0]*(ub[0] - lb[0])+(ub[0] + lb[0]))
        traf_g = 0.5*(y_pred[1][:,2]*(ub[2] - lb[2])+(ub[2] + lb[2]))
        traf_b = 0.5*(y_pred[1][:,1]*(ub[1] - lb[1])+(ub[1] + lb[1]))
        constraint = traf_a*K.square(traf_g)+traf_b         
        l2 = K.mean(K.square(y_pred[1] - y_true[1])) +weight_constraints*K.mean(1/(1+K.exp(-param*(constraint-1))))
        return l1+weight_losses
    return weighted_loss

"""
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
NNprice_Intrinsic.load_weights("intrinsic_price_rrmse_weights_1net_2000_moneynesss.h5")
for i in range(22):
    Autoencoder.layers[47-i].set_weights(NNprice_Intrinsic.layers[21-i].get_weights())

"""


Autoencoder.load_weights("autoencoder_weights.h5")

#Autoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.005,p2=15)],loss_weights =[1,10], optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=[["MAPE"],["MSE",miss_count]])
#history_Autoencoder = Autoencoder.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =25, verbose = True, shuffle=1)#,callbacks=[es])
#Autoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.002,p2=15)],loss_weights =[1,10], optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=[["MAPE"],["MSE",miss_count]])
#history_Autoencoder = Autoencoder.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =25, verbose = True, shuffle=1)#,callbacks=[es])
#Autoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.0015,p2=15)],loss_weights =[1,15], optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=[["MAPE"],["MSE",miss_count]])
#history_Autoencoder = Autoencoder.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =25, verbose = True, shuffle=1)#,callbacks=[es])
#Autoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.0015,p2=15)],loss_weights =[1,30], optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=[["MAPE"],["MSE",miss_count]])
#history_Autoencoder2 = Autoencoder.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =25, verbose = True, shuffle=1)#,callbacks=[es])
#Autoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.0015,p2=15)],loss_weights =[1,30], optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=[["MAPE"],["MSE",miss_count]])
#history_Autoencoder3 = Autoencoder.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =50, verbose = True, shuffle=1)#,callbacks=[es])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#Autoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.001,p2=15)],loss_weights =[100,3200], optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=[["MAPE"],["MSE",miss_count]])
#history_Autoencoder4 = Autoencoder.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =250, verbose = True, shuffle=1,callbacks=[es])
#checkpoint_filepath = '/tmp/checkpoint'
#cp = ModelCheckpoint(
#    filepath=checkpoint_filepath,
#    save_weights_only=True,
#    monitor='val_loss',
#    mode='min',
#    save_freq='epoch',
#    period = 10,
#    save_best_only=True)
#Autoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.0005,p2=15)],loss_weights =[100,4000], optimizer =Adam(clipvalue=10,learning_rate=5e-6),metrics=[["MAPE"],["MSE",miss_count]])
#history_Autoencoder5 = Autoencoder.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =50, verbose = True, shuffle=1,callbacks=[es,cp])
#Autoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.0002,p2=15)],loss_weights =[100,4000], optimizer =Adam(clipvalue=10,learning_rate=1e-6),metrics=[["MAPE"],["MSE",miss_count]])
#history_Autoencoder6 = Autoencoder.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =1000, verbose = True, shuffle=1,callbacks=[es,cp])
Autoencoder.save_weights("autoencoder_weights.h5")


Autoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.0002,p2=15)],loss_weights =[100,4000], optimizer =Adam(clipvalue=10,learning_rate=1e-6),metrics=[["MAPE"],["MSE",miss_count]])
history_Autoencoder7 = Autoencoder.fit(input_train, outputs_train, batch_size=250, validation_data = (input_val, outputs_val), epochs =1000, verbose = True, shuffle=1,callbacks=[es])

forecast_auto = Autoencoder.predict(input_test)

params_auto = np.array([myinverse(x) for x in forecast_auto[1]])
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios = calibration_plotter(params_auto,parameters[idx_test,:])
summary_calibration_auto = 100*np.asarray([np.quantile(error,0.05,axis=0),np.quantile(error,0.25,axis=0),np.median(error,axis=0),np.mean(error,axis=0),np.quantile(error,0.75,axis=0),np.quantile(error,0.95,axis=0),np.max(error,axis=0)])

prediction_intrinsic  = intrinsic_net[idx_test,:,:]+ forecast_auto[0].reshape((Ntest,Nmaturities,Nstrikes))
price_test           = price_1[idx_test,:,:]
vega_test            = vega_1[idx_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_intrinsic,price_test,vega_test)

NNprice_part = Sequential() 
NNprice_part.add(InputLayer(input_shape=(Nparameters+Nmaturities,1,1,)))
NNprice_part.add(ZeroPadding2D(padding=(2, 2)))
NNprice_part.add(Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NNprice_part.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_part.add(ZeroPadding2D(padding=(2,2)))
NNprice_part.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_part.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_part.add(ZeroPadding2D(padding=(2,2)))
NNprice_part.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_part.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_part.add(ZeroPadding2D(padding=(2,2)))
NNprice_part.add(Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_part.add(ZeroPadding2D(padding=(2,2)))
NNprice_part.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_part.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_part.add(ZeroPadding2D(padding=(2,2)))
NNprice_part.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_part.add(ZeroPadding2D(padding=(2,2)))
NNprice_part.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_part.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_part.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NNprice_part.add(Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NNprice_part.add(Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(S0/2,1,0,0)))#, kernel_constraint = tf.keras.constraints.NonNeg()))
for i in range(22):
    NNprice_part.layers[21-i].set_weights(Autoencoder.layers[47-i].get_weights())
testset = np.concatenate((parameters_trafo,rates),axis=1).reshape((Ntotal,Nparameters+Nmaturities,1,1))
testset = testset[idx_test,:]
prediction = NNprice_part.predict(testset)
prediction_intrinsic  = intrinsic_net[idx_test,:,:]+ prediction.reshape((Ntest,Nmaturities,Nstrikes))
price_test           = price_1[idx_test,:,:]
vega_test            = vega_1[idx_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_intrinsic,price_test,vega_test)



NNcalibrationP = Sequential() 
NNcalibrationP.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NNcalibrationP.add(ZeroPadding2D(padding=(1,1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(ZeroPadding2D(padding=(1,1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(ZeroPadding2D(padding=(1,1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(ZeroPadding2D(padding=(1,1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP.add(Flatten())
NNcalibrationP.add(Dense(Nparameters,activation = sig_scaled(2,1,0,-1),use_bias=True))
es = EarlyStopping(monitor='val_MSE', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
NNcalibrationP.compile(loss =log_constraint(param=1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib1 = NNcalibrationP.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP.compile(loss =log_constraint(param=0.1,p2=15), optimizer = "adam",metrics=["MAPE", "MSE",miss_count])
history_calib2 = NNcalibrationP.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP.compile(loss =log_constraint(param=0.02,p2=15), optimizer = Adam(learning_rate=1e-4),metrics=["MAPE", "MSE",miss_count])
history_calib3 = NNcalibrationP.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP.compile(loss =log_constraint(param=0.005,p2=15), optimizer = Adam(learning_rate=1e-4),metrics=["MAPE", "MSE",miss_count])
history_calib4 = NNcalibrationP.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP.compile(loss =log_constraint(param=0.0005,p2=15), optimizer = Adam(learning_rate=1e-4),metrics=["MAPE", "MSE",miss_count])
history_calib5 = NNcalibrationP.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP.compile(loss =log_constraint(param=0.0002,p2=15), optimizer = Adam(learning_rate=5e-5),metrics=["MAPE", "MSE",miss_count])
history_calib6 = NNcalibrationP.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP.compile(loss =log_constraint(param=0.00015,p2=15), optimizer = Adam(learning_rate=5e-6),metrics=["MAPE", "MSE",miss_count])
history_calib7 = NNcalibrationP.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP.compile(loss =log_constraint(param=0.00005,p2=15), optimizer = Adam(learning_rate=2e-6),metrics=["MAPE", "MSE",miss_count])
history_calib8 = NNcalibrationP.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=2000, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP.save_weights("calibrationweights_Price.h5")


prediction_calibration1 = NNcalibrationP.predict(input_test[0])
prediction_invtrafo1= np.array([myinverse(x) for x in prediction_calibration1])
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios = calibration_plotter(prediction_invtrafo1,np.array([myinverse(x) for x in outputs_test[1]]))
summary_calibration = 100*np.asarray([np.quantile(error,0.05,axis=0),np.quantile(error,0.25,axis=0),np.median(error,axis=0),np.mean(error,axis=0),np.quantile(error,0.75,axis=0),np.quantile(error,0.95,axis=0),np.max(error,axis=0)])

# In[InverseAutoEncoder]:

Inputpricer   = Input(shape=(Nparameters+Nmaturities,1,1,))
pricer2 = ZeroPadding2D(padding=(2,2))(Inputpricer)
pricer2 = Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu')(pricer2)
pricer2 = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer2)
pricer2 = ZeroPadding2D(padding=(2,2))(pricer2)
pricer2 = Conv2D(32, (2, 2), padding='valid',use_bias =True,strides =(1,1),activation='elu')(pricer2)
pricer2 = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer2)
pricer2 = ZeroPadding2D(padding=(2,2))(pricer2)
pricer2 = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer2)
pricer2 = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer2)
pricer2 = ZeroPadding2D(padding=(2,2))(pricer2)
pricer2 = Conv2D(32, (3,2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer2)
pricer2 = ZeroPadding2D(padding=(2,2))(pricer2)
pricer2 = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer2)
pricer2 = Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer2)
pricer2 = ZeroPadding2D(padding=(2,2))(pricer2)
pricer2 = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu')(pricer2)
pricer2 = ZeroPadding2D(padding=(2,2))(pricer2)
pricer2 = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu')(pricer2)
pricer2 = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu')(pricer2)
pricer2 = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(2,1),activation ='elu')(pricer2)
pricer2 = Conv2D(32, (2, 3),padding='valid',use_bias =True,strides =(1,1),activation ='elu')(pricer2)
pricer2 = Conv2D(9, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation =sig_scaled(S0/2,1,0,0))(pricer2)
pricer2 = Model(inputs = Inputpricer,outputs= pricer2)

calib2 = Reshape((Nmaturities,Nstrikes,1))(pricer2.output)
calib2 = ZeroPadding2D(padding=(1,1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = ZeroPadding2D(padding=(1,1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = ZeroPadding2D(padding=(1,1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = ZeroPadding2D(padding=(1,1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1))(calib2)
calib2 = Flatten()(calib2)
calib2 = Dense(Nparameters,activation = sig_scaled(2,1,0,-1),use_bias=True)(calib2)
InverseAutoencoder = Model(inputs =[Inputpricer],outputs=[pricer2.output,calib2])
input_invAutoencoder = np.concatenate((output_2,input_2),axis=1).reshape((Ntotal,Nparameters+Nmaturities,1,1,))

#pricer2.load_weights("intrinsic_price_rrmse_weights_1net_2000_moneynesss.h5")
#for i in range(22):
#    InverseAutoencoder.layers[45-i].set_weights(Autoencoder.layers[22-i].get_weights())

#InverseAutoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.01,p2=15)],loss_weights =[5,1], optimizer =Adam(clipvalue=10,learning_rate=1e-4),metrics=[["MAPE"],["MSE",miss_count]])
#history_InverseAutoencoder = InverseAutoencoder.fit(input_invAutoencoder[idx_train,:], outputs_train, batch_size=250, validation_data = (input_invAutoencoder[idx_val,:], outputs_val), epochs =25, verbose = True, shuffle=1)#,callbacks=[es])
#es = EarlyStopping(monitor='val_conv2d_154_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
#InverseAutoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.007,p2=15)],loss_weights =[100,20], optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=[["MAPE"],["MSE",miss_count]])
#history_InverseAutoencoder = InverseAutoencoder.fit(input_invAutoencoder[idx_train,:], outputs_train, batch_size=250, validation_data = (input_invAutoencoder[idx_val,:], outputs_val), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#InverseAutoencoder.compile(loss = [root_relative_mean_squared_error,log_constraint(param=0.005,p2=15)],loss_weights =[100,20], optimizer =Adam(clipvalue=10,learning_rate=2e-6),metrics=[["MAPE"],["MSE",miss_count]])
#history_InverseAutoencoder2 = InverseAutoencoder.fit(input_invAutoencoder[idx_train,:], outputs_train, batch_size=250, validation_data = (input_invAutoencoder[idx_val,:], outputs_val), epochs =1000, verbose = True, shuffle=1,callbacks=[es])
#InverseAutoencoder.save_weights("inverseautoencoder_weights.h5")
InverseAutoencoder.load_weights("inverseautoencoder_weights.h5")
#es = EarlyStopping(monitor='val_conv2d_154_loss', mode='min', verbose=1,patience = 20 ,restore_best_weights=True)
#InverseAutoencoder.compile(loss = ["MAPE",log_constraint(param=0.005,p2=15)],loss_weights =[1,50], optimizer =Adam(clipvalue=10,learning_rate=1e-5),metrics=[["MAPE"],["MSE",miss_count]])
#history_InverseAutoencoder3 = InverseAutoencoder.fit(input_invAutoencoder[idx_train,:], outputs_train, batch_size=500, validation_data = (input_invAutoencoder[idx_val,:], outputs_val), epochs =1000, verbose = True, shuffle=1,callbacks=[es])

prediction_invauto = InverseAutoencoder.predict(input_invAutoencoder[idx_test,:])
prediction_intrinsic  = intrinsic_net[idx_test,:,:]+ prediction_invauto[0].reshape((Ntest,Nmaturities,Nstrikes))
price_test           = price_1[idx_test,:,:]
vega_test            = vega_1[idx_test,:,:]
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_intrinsic,price_test,vega_test)

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
NNprice_Intrinsic.load_weights("intrinsic_price_rrmse_weights_1net_2000_moneynesss.h5")

prediction_intrinsic  = intrinsic_net[idx_test,:,:]+ NNprice_Intrinsic.predict(input_invAutoencoder[idx_test,:]).reshape((Ntest,Nmaturities,Nstrikes))
err_rel_mat,err_mat,err_optll,err_iv_approx,tmp,tmp= pricing_plotter(prediction_intrinsic,price_test,vega_test)


# In[Testing Area]:
NNcalibrationP2 = Sequential() 
NNcalibrationP2.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NNcalibrationP2.add(ZeroPadding2D(padding=(1,1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True, padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(ZeroPadding2D(padding=(1,1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(ZeroPadding2D(padding=(2,2)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(ZeroPadding2D(padding=(2,2)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation =sig_scaled(2,1,0,-1)))
NNcalibrationP2.add(Flatten())
NNcalibrationP2.add(Dense(Nparameters,activation = sig_scaled(2,1,0,-1),use_bias=True))
NNcalibrationP2.summary()
es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)
NNcalibrationP2.compile(loss =log_constraint(param=1,p2=15), optimizer =  Adam(clipvalue=10),metrics=["MAPE", "MSE",miss_count])
history2_calib1 = NNcalibrationP2.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP2.compile(loss =log_constraint(param=0.1,p2=15), optimizer =  Adam(clipvalue=10),metrics=["MAPE", "MSE",miss_count])
history2_calib2 = NNcalibrationP2.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP2.compile(loss =log_constraint(param=0.02,p2=15), optimizer = Adam(learning_rate=1e-4,clipvalue=10),metrics=["MAPE", "MSE",miss_count])
history2_calib3 = NNcalibrationP2.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP2.compile(loss =log_constraint(param=0.005,p2=15), optimizer = Adam(learning_rate=1e-4,clipvalue=10),metrics=["MAPE", "MSE",miss_count])
history2_calib4 = NNcalibrationP2.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP2.compile(loss =log_constraint(param=0.0005,p2=15), optimizer = Adam(learning_rate=1e-4,clipvalue=10),metrics=["MAPE", "MSE",miss_count])
history2_calib5 = NNcalibrationP2.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP2.compile(loss =log_constraint(param=0.0002,p2=15), optimizer = Adam(learning_rate=5e-5,clipvalue=10),metrics=["MAPE", "MSE",miss_count])
history2_calib6 = NNcalibrationP2.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP2.compile(loss =log_constraint(param=0.00015,p2=15), optimizer = Adam(learning_rate=5e-6,clipvalue=10),metrics=["MAPE", "MSE",miss_count])
history2_calib7 = NNcalibrationP2.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=5, verbose = True, shuffle=1,callbacks =[es])
NNcalibrationP2.compile(loss =log_constraint(param=0.00005,p2=15), optimizer = Adam(learning_rate=2e-6,clipvalue=10),metrics=["MAPE", "MSE",miss_count])
history2_calib8 = NNcalibrationP2.fit(input_train[0],outputs_train[1], batch_size=250, validation_data = (input_val[0],outputs_val[1]), epochs=2000, verbose = True, shuffle=1,callbacks =[es])


prediction_calibration1 = NNcalibrationP.predict(input_test[0])
prediction_invtrafo1= np.array([myinverse(x) for x in prediction_calibration1])
error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios = calibration_plotter(prediction_invtrafo1,np.array([myinverse(x) for x in outputs_test[1]]))
summary_calibration = 100*np.asarray([np.quantile(error,0.05,axis=0),np.quantile(error,0.25,axis=0),np.median(error,axis=0),np.mean(error,axis=0),np.quantile(error,0.75,axis=0),np.quantile(error,0.95,axis=0),np.max(error,axis=0)])
