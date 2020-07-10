import numpy as np
from tensorflow.compat.v1.keras.models import Sequential,Model
from tensorflow.compat.v1.keras.layers import Reshape,InputLayer,Dense,Flatten, Conv2D,Conv1D, Dropout, Input,ZeroPadding2D,ZeroPadding1D,MaxPooling2D
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.python.client import device_lib
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)
print(device_lib.list_local_devices())
tf.compat.v1.keras.backend.set_floatx('float64')  

def root_relative_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_pred-y_true)/y_true)))   
def mse_constraint(param):
    def rel_mse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            return K.mean(K.square(y_pred - y_true)) +param*K.mean(K.greater(constraint,1))
    return rel_mse_constraint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 50 ,restore_best_weights=True)

# Volatility Pricing Network   
NN1a = Sequential() 
NN1a.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1a.add(ZeroPadding2D(padding=(2, 2)))
NN1a.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NN1a.add(ZeroPadding2D(padding=(3,1)))
NN1a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1a.add(ZeroPadding2D(padding=(2,2)))
NN1a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1a.add(ZeroPadding2D(padding=(1,1)))
NN1a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1a.add(ZeroPadding2D(padding=(1,1)))
NN1a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1a.add(ZeroPadding2D(padding=(1,1)))
NN1a.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1a.add(ZeroPadding2D(padding=(2,1)))
NN1a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1a.add(ZeroPadding2D(padding=(2,1)))
NN1a.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1a.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
NN1a.summary()
NN1a.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN1b = Sequential() 
NN1b.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1b.add(ZeroPadding2D(padding=(2, 2)))
NN1b.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))
NN1b.add(ZeroPadding2D(padding=(3,1)))
NN1b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1b.add(ZeroPadding2D(padding=(2,1)))
NN1b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1b.add(ZeroPadding2D(padding=(1,0)))
NN1b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1b.add(ZeroPadding2D(padding=(1,1)))
NN1b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1b.add(ZeroPadding2D(padding=(1,1)))
NN1b.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,2),activation ='elu'))
NN1b.add(ZeroPadding2D(padding=(2,1)))
NN1b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1b.add(ZeroPadding2D(padding=(2,1)))
NN1b.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1b.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='linear', kernel_constraint = tf.keras.constraints.NonNeg()))
NN1b.summary()
NN1b.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])

y_tmp_train = y_train[:,:,[0,1,2,3,4,5],:]
y_tmp_val = y_val[:,:,[0,1,2,3,4,5],:]
y_tmp_test= y_test[:,:,[0,1,2,3,4,5],:]
NN1a.fit(X_train, y_tmp_train, batch_size=64, validation_data = (X_val, y_tmp_val), epochs = 1000, verbose = True, shuffle=1,callbacks=[es])
y_tmp_train = y_train_[:,:,[5,6,7,8],:]
y_tmp_val = y_val[:,:,[5,6,7,8],:]
y_tmp_test= y_test[:,:,[5,6,7,8],:]
NN1b.fit(X_train, y_tmp_train, batch_size=64, validation_data = (X_val, y_tmp_val), epochs = 1000, verbose = True, shuffle=1,callbacks=[es])
prediction_a   = NN1a.predict(X_test_trafo).reshape((Ntest,6,Nstrikes))
prediction_b   = NN1b.predict(X_test_trafo).reshape((Ntest,4,Nstrikes))
prediction = np.concatenate((prediction_a[:,[0,1,2,3,4],:],prediction_b[:,[0,1,2,3],:]),axis =1)

# Calibration Network
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
NN2.compile(loss =mse_constraint(0.75), optimizer = "adam",metrics=["MAPE", "MSE"])
NN2.fit(y_train,X_train, batch_size=50, validation_data = (y_val,X_val), epochs=1000, verbose = True, shuffle=1,callbacks = [es])



