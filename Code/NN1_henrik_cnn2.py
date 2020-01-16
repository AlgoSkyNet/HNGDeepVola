# Neuronal Network 1 for learning the implied vola 
# source: https://github.com/amuguruza/NN-StochVol-Calibrations/blob/master/1Factor/Flat%20Forward%20Variance/NN1Factor.ipynb 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import InputLayer,Dense,Dropout,Conv2D,MaxPooling2D,ZeroPadding2D,Flatten
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import py_vollib.black_scholes.implied_volatility as vol
import time
import scipy
import scipy.io
import tensorflow as tf

# Data Import

mat         = scipy.io.loadmat('data_vola_maxbounds_50000_0005_09_11_30_210.mat')
data        = mat['data_vola']
Nparameters = 5
maturities  = np.array([30, 60, 90, 120, 150, 180, 210])
strikes     = np.array([0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1])
Nstrikes    = len(strikes)   
Nmaturities = len(maturities)   
xx          = data[:,:Nparameters]
yy          = data[:,Nparameters+2:]


# split into train and test sample
X_train, X_test, y_train, y_test = train_test_split(
    xx, yy, test_size=0.15)#, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.15)#, random_state=42)

Ntest= X_test.shape[0]
Ntrain= X_train.shape[0]
Nval= X_val.shape[0]
scale=StandardScaler()
y_train_transform = scale.fit_transform(y_train)
y_val_transform   = scale.transform(y_val)
y_test_transform  = scale.transform(y_test)
def ytransform(y_train,y_val,y_test):
    #return [scale.transform(y_train),scale.transform(y_val), 
    #        scale.transform(y_test)]
    return [y_train,y_val,y_test]
def yinversetransform(y):
    return y
    #return scale.inverse_transform(y)
[y_train_trafo, y_val_trafo, y_test_trafo]=ytransform(y_train, y_val, y_test)
y_train_trafo = np.asarray([y_train[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
y_val_trafo =  np.asarray([y_val[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])
y_test_trafo =  np.asarray([y_test[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])
ub=np.amax(xx, axis=0)
lb=np.amin(xx, axis=0)
def myscale(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=(x[i] - (ub[i] + lb[i])*0.5) * 2 / (ub[i] - lb[i])
    return res
def myinverse(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=x[i]*(ub[i] - lb[i]) *0.5 + (ub[i] + lb[i])*0.5
    return res

X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo   = np.array([myscale(x) for x in X_val])
X_test_trafo  = np.array([myscale(x) for x in X_test])
X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo   = np.array([myscale(x) for x in X_val])
X_test_trafo  = X_test_trafo.reshape((Ntest,5,1,1))
X_train_trafo = X_train_trafo.reshape((Ntrain,5,1,1))
X_val_trafo   = X_val_trafo.reshape((Nval,5,1,1))

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def root_relative_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))    
def root_relative_mean_squared_error_convexity(param):
    def help_error(y_true, y_pred):
        error = K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))
        for i in range(1,Nmaturities-1):
            for j in range(1,Nstrikes-1):
                convexity = K.mean(K.control_flow_ops.math_ops.logical_or(K.greater(y_pred[:,0,i,j],0.5*(y_pred[:,0,i-1,j]+y_pred[:,0,i+1,j])),K.greater(y_pred[:,0,i,j],0.5*(y_pred[:,0,i,j-1]+y_pred[:,0,i,j+1]))))       
                error = error+param*convexity/((Nstrikes-1)*(Nmaturities-1))
        return error
    return help_error
#Neural Network
keras.backend.set_floatx('float64')
saver = np.zeros(np.arange(0,0.6,0.05).shape)
count = 0
for param in np.arange(0,0.6,0.05):
    NN1 = Sequential() 
    NN1.add(InputLayer(input_shape=(Nparameters,1,1,)))
    NN1.add(ZeroPadding2D(padding=(2, 2)))
    NN1.add(Conv2D(32, (3, 1), padding='valid',strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
    NN1.add(ZeroPadding2D(padding=(1,1)))
    NN1.add(Conv2D(32, (2, 2),padding='valid',strides =(1,1),activation ='elu'))
    NN1.add(Conv2D(32, (2, 2),padding='valid',strides =(2,1),activation ='elu'))
    NN1.add(ZeroPadding2D(padding=(1,1)))
    NN1.add(Conv2D(32, (3,3),padding='valid',strides =(2,1),activation ='elu'))
    NN1.add(ZeroPadding2D(padding=(1,1)))
    
    NN1.add(Conv2D(32, (2, 2),padding='valid',strides =(2,1),activation ='elu'))
    NN1.add(ZeroPadding2D(padding=(1,1)))
    
    NN1.add(Conv2D(32, (2, 2),padding='valid',strides =(2,1),activation ='elu'))
    #NN1.add(MaxPooling2D(pool_size=(2, 1)))
    #NN1.add(Dropout(0.25))
    #NN1.add(ZeroPadding2D(padding=(0,1)))
    NN1.add(Conv2D(Nstrikes, (2, 1),padding='valid',strides =(2,1),activation ='linear', kernel_constraint = keras.constraints.NonNeg()))
    #NN1.add(MaxPooling2D(pool_size=(4, 1)))
    #NN1.summary()
    
    
                    
    NN1.compile(loss = root_relative_mean_squared_error_convexity(param), optimizer = "adam",metrics=["MAPE","MSE"])
    #NN1.compile(loss = "mean_squared_error", optimizer = "adam",metrics=["MAPE"])
    #NN1.compile(loss = root_relative_mean_squared_error_lasso, optimizer = "adam",metrics=[root_relative_mean_squared_error,"mean_squared_error"])
    #NN1.compile(loss = 'mean_absolute_percentage_error', optimizer = "adam")
    NN1.fit(X_train_trafo, y_train_trafo, batch_size=64, validation_data = (X_val_trafo, y_val_trafo),
            epochs = 30, verbose = True, shuffle=1)
    #NN1.save_weights('NN_HNGarch_weights.h5')
    saver[count] = NN1.evaluate(X_val_trafo, y_val_trafo)[1]
    print(count)
    count += 1 
#NN1.compile(loss = root_relative_mean_squared_error_lasso, optimizer = "adam",metrics=[root_relative_mean_squared_error,"mean_squared_error"])
#NN1.compile(loss = 'mean_absolute_percentage_error', optimizer = "adam")

plt.figure(figsize=(14,4))
plt.scatter(np.arange(0,0.6,0.05),saver)
plt.xlabel("constraint parameter",fontsize=15,labelpad=5)
plt.ylabel("validation error",fontsize=15,labelpad=5)
plt.show()
count = 0
error = np.zeros(np.arange(0,0.6,0.05).shape)
def convex_violation(y_pred):
    error = np.zeros(y_pred.shape[0])
    for i in range(Nmaturities):
        if (i==0 or i==Nmaturities-1):
            for j in range(1,Nstrikes-1):
                convexity = y_pred[:,0,i,j]>0.5*(y_pred[:,0,i,j-1]+y_pred[:,0,i,j+1])       
                error +=   convexity

        else: 
            for j in range(1,Nstrikes-1):
                convexity = np.logical_or(y_pred[:,0,i,j]>0.5*(y_pred[:,0,i-1,j]+y_pred[:,0,i+1,j]),y_pred[:,0,i,j]>0.5*(y_pred[:,0,i,j-1]+y_pred[:,0,i,j+1]))       
                error +=   convexity

    for j in [0,Nstrikes-1]:
        for i in range(1,Nmaturities-1):
            convexity = y_pred[:,0,i,j]>0.5*(y_pred[:,0,i-1,j]+y_pred[:,0,i+1,j])     
            error +=   convexity
    mean_error = np.mean(error)
    return  error,mean_error    
       

#==============================================================================
#error plots
S0=1.
y_test_re       = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
prediction = NN1.predict(X_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
err_rel_mat  = np.zeros(prediction.shape)
err_mat  =np.zeros(prediction.shape)
for i in range(Ntest):
    err_rel_mat[i,:,:] =  np.abs((y_test_re[i,:,:]-prediction[i,:,:])/y_test_re[i,:,:])
    err_mat[i,:,:] =  np.square((y_test_re[i,:,:]-prediction[i,:,:]))
idx = np.argsort(np.max(err_rel_mat,axis=tuple([1,2])), axis=None)
#bad_idx = idx[:-200]
bad_idx = idx
#from matplotlib.colors import LogNorm
plt.figure(figsize=(14,4))
ax=plt.subplot(2,3,1)
err1 = 100*np.mean(err_rel_mat[bad_idx,:,:],axis=0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,2)
err2 = 100*np.std(err_rel_mat[bad_idx,:,:],axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,3)
err3 = 100*np.max(err_rel_mat[bad_idx,:,:],axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,4)
err1 = np.sqrt(np.mean(err_mat[bad_idx,:,:],axis=0))
plt.title("RMSE",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,5)
err2 = np.std(err_mat[bad_idx,:,:],axis = 0)
plt.title("Std MSE",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,6)
err3 = np.max(err_mat[bad_idx,:,:],axis = 0)
plt.title("Maximum MSE",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.show()
plt.savefig('HNG_NNErrors_Cnn2.png', dpi=300)




#==============================================================================
#surface
import random
#test_sample = random.randint(0,len(y_test))
test_sample = idx[-1] #worst case
y_predict_sample_p = prediction[test_sample,:,:]
y_test_sample_p = y_test_re[test_sample,:,:]
diff = y_test_sample_p-y_predict_sample_p 
rel_diff = np.abs(y_test_sample_p-y_predict_sample_p)/(y_test_sample_p)
    
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = strikes
Y = maturities
X, Y = np.meshgrid(X, Y)


#ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.plot_surface(X, Y, y_test_sample_p, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, y_predict_sample_p , rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
#ax.plot_surface(X, Y, rel_diff, rstride=1, cstride=1,
#                cmap='viridis', edgecolor='none')
ax.set_xlabel('Strikes')
ax.set_ylabel('Maturities')
#ax.set_zlabel('rel. err');
plt.show()


#==============================================================================
#smile
sample_ind = test_sample
X_sample = X_test_trafo[sample_ind,:,:].reshape((Nparameters,))
y_sample = y_test[sample_ind]
#print(scale.inverse_transform(y_sample))

prediction_sample=NN1.predict(X_sample.reshape(1,5,1,1)).reshape((Nstrikes*Nmaturities,))
plt.figure(figsize=(14,12))
for i in range(Nmaturities):
    plt.subplot(4,4,i+1)
    
    plt.plot(np.log(strikes/S0),y_sample[i*Nstrikes:(i+1)*Nstrikes],'b',label="Input data")
    plt.plot(np.log(strikes/S0),prediction_sample[i*Nstrikes:(i+1)*Nstrikes],'--r',label=" NN Approx")
    #plt.ylim(0.22, 0.26)
    
    plt.title("Maturity=%1.2f "%maturities[i])
    plt.xlabel("log-moneyness")
    plt.ylabel("Implied vol")
    
    plt.legend()
plt.tight_layout()
plt.savefig('HNG_smile_cnn2.png', dpi=300)
plt.show()

#==============================================================================
# gradient methods for optimization with Levenberg-Marquardt
NNParameters=[]
for i in range(0,len(NN1.layers)):
    NNParameters.append(NN1.layers[i].get_weights())

NumLayers=3
def elu(x):
    #Careful function ovewrites x
    ind=(x<0)
    x[ind]=np.exp(x[ind])-1
    return x
def eluPrime(y):
    # we make a deep copy of input x
    x=np.copy(y)
    ind=(x<0)
    x[ind]=np.exp(x[ind])
    x[~ind]=1
    return x
def NeuralNetworkGradient(x):
    input1=x
    #Identity Matrix represents Jacobian with respect to initial parameters
    grad=np.eye(Nparameters)
    #Propagate the gradient via chain rule
    for i in range(NumLayers):
        input1=(np.dot(input1,NNParameters[i][0])+NNParameters[i][1])
        grad=(np.einsum('ij,jk->ik', grad, NNParameters[i][0]))
        #Elu activation
        grad*=eluPrime(input1)
        input1=elu(input1)
    #input1.append(np.dot(input1[i],NNParameters[i+1][0])+NNParameters[i+1][1])
    grad=np.einsum('ij,jk->ik',grad,NNParameters[i+1][0])
    #grad stores all intermediate Jacobians, however only the last one is used here as output
    return grad

#Cost Function for Levenberg Marquardt
def CostFuncLS(x,sample_ind):
    return (NN1.predict(x.reshape(1,Nparameters))[0]-y_test_trafo[sample_ind])
def JacobianLS(x,sample_ind):
    return NeuralNetworkGradient(x).T

def CostFunc(x,sample_ind):
    return np.sum(np.power((NN1.predict(x.reshape(1,Nparameters))[0]-y_test_trafo[sample_ind]),2))
def Jacobian(x,sample_ind):
    return 2*np.sum((NN1.predict(x.reshape(1,Nparameters))[0]-y_test_trafo[sample_ind])*NeuralNetworkGradient(x),axis=1)

Approx=[]
Timing=[]
#sample_ind = 500
#X_sample = X_test_trafo[sample_ind]
#y_sample = y_test[sample_ind]
solutions=np.zeros([1,Nparameters])
#times=np.zeros(1)
init=np.zeros(Nparameters)
n = X_test.shape[0]
#n=1000

for i in range(n):
    disp=str(i+1)+"/"+str(n)
    print (disp,end="\r")
    #L-BFGS-B
    #start= time.clock()
    #I=scipy.optimize.minimize(CostFunc,x0=init,args=i,method='L-BFGS-B',jac=Jacobian,tol=1E-10,options={"maxiter":5000})
    #end= time.clock()
    #solutions=myinverse(I.x)
    #times=end-start
    #Levenberg-Marquardt
    start= time.clock()
    I=scipy.optimize.least_squares(CostFuncLS, init, JacobianLS, args=(i,), gtol=1E-10)
    end= time.clock()
    solutions=myinverse(I.x)
    times=end-start
    Approx.append(np.copy(solutions))
    Timing.append(np.copy(times))
LMParameters=[Approx[i] for i in range(len(Approx))]
#np.savetxt("NNParametersHNG_cnn.txt",LMParameters)

#==============================================================================
#Calibration Errors with Levenberg-Marquardt¶
titles=["$\\alpha$","$\\beta$","$\\gamma$","$\\omega$", "$\\sigma$"]
average=np.zeros([Nparameters,n])
fig=plt.figure(figsize=(12,8))
for u in range(Nparameters):
    ax=plt.subplot(2,3,u+1)
    for i in range(n):
        
        X=X_test[i][u]
        plt.plot(X,100*np.abs(LMParameters[i][u]-X)/np.abs(X),'b*')
        average[u,i]=np.abs(LMParameters[i][u]-X)/np.abs(X)
    plt.title(titles[u],fontsize=20)
    plt.ylabel('relative Error',fontsize=15)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter() )
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.text(0.5, 0.8, 'Average: %1.2f%%\n Median:   %1.2f%% '%(np.mean(100*average[u,:]),
                                                                np.quantile(100*average[u,:],0.5)), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=15)

    print("average= ",np.mean(average[u,:]))
plt.tight_layout()
plt.savefig('HNG_ParameterRelativeErrors_cnn.png', dpi=300)
plt.show()



#==============================================================================
#RMSE plot
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * scipy.stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2, 0.0, 1.0))
    return call

RMSE_opt = np.zeros(n)
RMSE_bs = np.zeros(n)
bs_prices = np.zeros((n,Nmaturities*Nstrikes))
Y = len(y_test[0,:])
for i in range(n):
    Y = y_test[i,:]
    Y_pred = yinversetransform(NN1.predict(myscale(LMParameters[i]).reshape(1,Nparameters))[0])
    RMSE_opt[i] = np.sqrt(np.mean((Y-Y_pred)**2))
    bs = np.zeros((Nmaturities,Nstrikes))
    for t in range(Nmaturities):
        for k in range(Nstrikes):
            bs[t,k] = bs_call(1,strikes[k],maturities[t]/252,0.005,np.sqrt(252*LMParameters[i][-1]))
    bs_prices[i,:] = bs.T.reshape((1,Nmaturities*Nstrikes))
    RMSE_bs[i] = np.sqrt(np.mean((Y-bs_prices[i,:])**2))
    
fig =plt.figure(figsize=(12,8)) 
plt.plot(RMSE_opt)
plt.yscale('log')
plt.plot(RMSE_bs)
plt.yscale('log')
plt.legend(['Neural Net','Black-Scholes'],fontsize=17)
plt.ylabel('RMSE')
plt.xlabel('Scenario')
plt.title("RMSE of optimal parameters",fontsize=20)
plt.savefig('RMSE_BS_Net_cnn.png', dpi=300)
    
fig=plt.figure(figsize=(12,8))
plt.plot(RMSE_opt,'b*')
plt.title("RMSE of optimal parameters",fontsize=20)
plt.ylabel('RMSE vola surface',fontsize=15)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter() )
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.text(0.5, 0.8, 'Average: %1.2f%%\n Median:   %1.2f%% '%(np.mean(100*RMSE_opt),
            np.quantile(100*RMSE_opt,0.5)), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=15)
plt.tight_layout()
plt.show()

#==============================================================================
titles=["$\\alpha$","$\\beta$","$\\gamma$","$\\omega$", "$\\sigma$"]
plt.figure(figsize=(18, 5))
plt.clf()
plt.subplot(121)

ax = plt.gca()
q=np.linspace(0,0.99,200)
for u in range(Nparameters):
    p=plt.plot(100*q,np.quantile(100*average[u,:],q),label=titles[u])
    
    c=p[0].get_color()
ymin, ymax = ax.get_ylim()
ax.set_xlim(0,100)
plt.plot(100*np.ones(2)*0.95,np.array([0,ymax]),'--k',label="95% quantile")
plt.title("Empirical CDF of parameter relative error",fontsize=20)
plt.legend(fontsize=17)
plt.xlabel("quantiles",fontsize=17)
plt.ylabel("relative error",fontsize=17)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter() )
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter() )
plt.tick_params(axis='both', which='major', labelsize=17)
plt.tick_params(axis='both', which='minor', labelsize=17)
plt.xticks(np.arange(0, 101, step=10))

plt.grid()
plt.subplot(122)


ax = plt.gca()
q=np.linspace(0,1,200)
p=plt.plot(100*q,np.quantile(100*RMSE_opt,q),linewidth=3,label="RMSE")
ymin, ymax = ax.get_ylim()
plt.plot(100*np.ones(2)*0.99,np.array([0,ymax]),'--k',label="99% quantile")
plt.title("Empirical CDF of implied vol surface RMSE",fontsize=20)
plt.legend(fontsize=17)
plt.xlabel("quantiles",fontsize=17)
plt.ylabel("RMSE",fontsize=17)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter() )
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter() )
plt.tick_params(axis='both', which='major', labelsize=17)
plt.tick_params(axis='both', which='minor', labelsize=17)
plt.xticks(np.arange(0, 101, step=10))
plt.grid()
plt.tight_layout()
plt.savefig('HNG_ErrorCDF_cnn.png', dpi=300)
plt.show()

#==============================================================================
#real data
sp500_mat = scipy.io.loadmat('surface_sp500_new.mat')
surface = sp500_mat['vector_surface']
surface_trafo = ytransform(surface,surface,surface)[2]
def CostFuncLS_real(x):
    return (yinversetransform(NN1.predict(x.reshape(1,Nparameters))[0])-surface_trafo).T.flatten()
def JacobianLS_real(x):
    return NeuralNetworkGradient(x).T

solutions=np.zeros([1,Nparameters])
init=np.zeros(Nparameters)

#Levenberg-Marquardt
I=scipy.optimize.least_squares(CostFuncLS_real, init, JacobianLS_real, gtol=1E-10)
LMParameters_real=myinverse(I.x)
Y_pred_real = yinversetransform(NN1.predict(myscale(LMParameters_real).reshape(1,Nparameters))[0])
RMSE_opt_real = np.sqrt(np.mean((surface-Y_pred_real)**2))
err_real_mean = np.mean(100*np.abs((surface-Y_pred_real)/surface),axis = 1)
err_real = (100*np.abs((surface-Y_pred_real)/surface)).reshape((Nmaturities, Nstrikes))
diff = (Y_pred_real - surface).reshape((Nmaturities, Nstrikes))
np.savetxt("SP500_pred_cNN.txt",Y_pred_real)