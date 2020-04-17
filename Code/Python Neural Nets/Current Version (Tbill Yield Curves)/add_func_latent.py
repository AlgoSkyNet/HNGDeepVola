### additional functions for main file
import numpy as np
from tensorflow.keras import backend as K
from config_latent import Nparameters,diff,bound_sum,ub,lb,Ntest,Nstrikes,strikes,Nmaturities,maturities
from config_latent import myscale,myinverse,ytransform,yinversetransform

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from py_vollib.black_scholes.implied_volatility import implied_volatility as bsimpvola
from py_vollib.black_scholes.implied_volatility import black_scholes
import os as os
from multiprocessing import Pool
import random
#import matplotlib.lines as mlines
#import matplotlib.transforms as mtransforms
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mpl_toolkits.mplot3d import Axes3D  
#from matplotlib import cm
import cmath
import math


### custom errors
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))   
    
def root_relative_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))   

def rmse_constraint(param):
    def rel_mse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))  +param*K.mean(K.greater(constraint,1))
    return rel_mse_constraint

def mse_constraint(param):
    def rel_mse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.mean(K.square(y_pred - y_true)) +param*K.mean(K.greater(constraint,1))
    return rel_mse_constraint


### constraints
def constraint_violation(x):
    return np.sum(x[:,0]*x[:,2]**2+x[:,1]>=1)/x.shape[0],x[:,0]*x[:,2]**2+x[:,1]>=1,x[:,0]*x[:,2]**2+x[:,1]

### error plot

# error pricer
def pricing_plotter(prediction,y_test):     
    err_rel_mat  = np.zeros(prediction.shape)
    err_mat      = np.zeros(prediction.shape)
    for i in range(y_test.shape[0]):
        err_rel_mat[i,:,:] =  np.abs((y_test[i,:,:]-prediction[i,:,:])/y_test[i,:,:])
        err_mat[i,:,:] =  np.square((y_test[i,:,:]-prediction[i,:,:]))
    idx = np.argsort(np.max(err_rel_mat,axis=tuple([1,2])), axis=None)
    
    #bad_idx = idx[:-200]
    bad_idx = idx
    #from matplotlib.colors import LogNorm
    plt.figure(figsize=(14,4))
    plt.suptitle('Errors Neural Net Pricing', fontsize=16)
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
    return err_rel_mat,err_mat,idx,bad_idx

#errror calibration
def calibration_plotter(prediction,X_test_trafo2,X_test):
    prediction_invtrafo= np.array([myinverse(x) for x in prediction])
    prediction_std = np.std(prediction,axis=0)
    error = np.zeros((Ntest,Nparameters))
    for i in range(Ntest):
        #error[i,:] =  np.abs((X_test_trafo2[i,:]-prediction[i,:])/X_test_trafo2[i,:])
        error[i,:] =  np.abs((X_test[i,:]-prediction_invtrafo[i,:])/X_test[i,:])
    err1 = np.mean(error,axis = 0)
    err2 = np.median(error,axis = 0)
    err_std = np.std(error,axis = 0)
    idx = np.argsort(error[:,0], axis=None)
    good_idx = idx[:-100]
        
    _,_,c =constraint_violation(prediction_invtrafo)
    _,_,c2 =constraint_violation(X_test)
    
    
    testing_violation = c>=1
    testing_violation2 = (c<1)
    vio_error = error[testing_violation,:]
    vio_error2 = error[testing_violation2,:]
    
    
    plt.figure(figsize=(14,4))
    plt.suptitle('Errors Neural Net Calibration', fontsize=16)
    ax=plt.subplot(1,3,1)
    plt.boxplot(error)
    plt.yscale("log")
    plt.xticks([1, 2, 3,4], ['w','a','b','g*'])
    plt.ylabel("Errors")
    ax=plt.subplot(1,3,2)
    plt.boxplot(vio_error)
    plt.yscale("log")
    plt.xticks([1, 2, 3,4], ['w','a','b','g*'])
    plt.ylabel("Errors parameter violation")
    ax=plt.subplot(1,3,3)
    plt.boxplot(vio_error2)
    plt.yscale("log")
    plt.xticks([1, 2, 3,4], ['w','a','b','g*'])
    plt.ylabel("Errors no parameter violation")
    plt.show()
    
    print("violation error mean in %:",100*np.mean(vio_error,axis=0))
    print("no violation error mean in %:",100*np.mean(vio_error2,axis=0))
    print("violation error median in %:",100*np.median(vio_error,axis=0))
    print("no violation error median in %:",100*np.median(vio_error2,axis=0))
    print("error mean in %:",100*err1)
    print("error median in %:",100*err2)
    
    fig = plt.figure()
    plt.suptitle('Errors Neural Net Calibration', fontsize=16)
    plt.scatter(c2,c)
    plt.plot(np.arange(0, 1.1,0.5),np.arange(0,1.1,0.5),'-r')
    plt.xlabel("True Constraint")
    plt.ylabel("Forecasted Constraint")
    
    
    plt.figure(figsize=(14,4))
    plt.suptitle('Errors Neural Net Calibration', fontsize=16)
    ax=plt.subplot(1,4,1)
    plt.yscale("log")
    plt.scatter(c2[testing_violation2],vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
    plt.scatter(c2[testing_violation],vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
    plt.xlabel("True Constraint")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,4,2)
    plt.yscale("log")
    plt.scatter(c2[testing_violation2],vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
    plt.scatter(c2[testing_violation],vio_error[:,1],c="b",s=1,marker="x",label="beta con")
    plt.xlabel("True Constraint")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,4,3)
    plt.yscale("log")
    plt.scatter(c2[testing_violation2],vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
    plt.scatter(c2[testing_violation],vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
    plt.xlabel("True Constraint")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,4,4)
    plt.yscale("log")
    plt.scatter(c2[testing_violation2],vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
    plt.scatter(c2[testing_violation],vio_error[:,3],c="b",s=1,marker="x",label="omega con")
    plt.xlabel("True Constraint")
    plt.ylabel("Relative Deviation")
    plt.legend()

    
    fig = plt.figure()
    plt.suptitle('Errors Neural Net Calibration', fontsize=16)
    plt.scatter(c2,c)
    plt.plot(np.arange(0, 1.1,0.5),np.arange(0, 1.1,0.5),'-r')
    plt.xlabel("True Constraint")
    plt.ylabel("Forecasted Constraint")
    
    plt.figure(figsize=(14,4))
    plt.suptitle('Errors Neural Net Calibration', fontsize=16)
    ax=plt.subplot(1,4,1)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
    plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
    plt.xlabel("constraint deviation")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,4,2)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
    plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,1],c="b",s=1,marker="x",label="beta con")
    plt.xlabel("constraint deviation")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,4,3)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
    plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
    plt.xlabel("constraint deviation")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,4,4)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
    plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,3],c="b",s=1,marker="x",label="omega con")
    plt.xlabel("constraint deviation")
    plt.ylabel("Relative Deviation")
    plt.legend()
    return error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2

def calibration_plotter_deterministic(prediction,X_test):
    prediction_std = np.std(prediction,axis=0)
    error = np.zeros((Ntest,Nparameters))
    for i in range(Ntest):
         error[i,:] =  np.abs((X_test[i,:]-prediction[i,:])/X_test[i,:])
    err1 = np.mean(error,axis = 0)
    err2 = np.median(error,axis = 0)
    err_std = np.std(error,axis = 0)
    idx = np.argsort(error[:,0], axis=None)
    good_idx = idx[:-100]
        
    _,_,c =constraint_violation(prediction)
    _,_,c2 =constraint_violation(X_test)
    
    
    testing_violation = c>=1
    testing_violation2 = (c<1)
    vio_error = error[testing_violation,:]
    vio_error2 = error[testing_violation2,:]
    
    
    plt.figure(figsize=(14,4))
    plt.suptitle('Errors Deterministic Calibration', fontsize=16)
    ax=plt.subplot(1,3,1)
    plt.boxplot(error)
    plt.yscale("log")
    plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
    plt.ylabel("Errors")
    ax=plt.subplot(1,3,2)
    plt.boxplot(vio_error)
    plt.yscale("log")
    plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
    plt.ylabel("Errors parameter violation")
    ax=plt.subplot(1,3,3)
    plt.boxplot(vio_error2)
    plt.yscale("log")
    plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
    plt.ylabel("Errors no parameter violation")
    plt.show()
    
    print("violation error mean in %:",100*np.mean(vio_error,axis=0))
    print("no violation error mean in %:",100*np.mean(vio_error2,axis=0))
    print("violation error median in %:",100*np.median(vio_error,axis=0))
    print("no violation error median in %:",100*np.median(vio_error2,axis=0))
    print("error mean in %:",100*err1)
    print("error median in %:",100*err2)
    
    fig = plt.figure()
    plt.suptitle('Errors Deterministic Calibration', fontsize=16)
    plt.scatter(c2,c)
    plt.plot(np.arange(0, 1.1,0.5),np.arange(0, 1.1,0.5),'-r')
    plt.xlabel("True Constraint")
    plt.ylabel("Forecasted Constraint")
    
    
    plt.figure(figsize=(14,4))
    plt.suptitle('Errors Deterministic Calibration', fontsize=16)
    ax=plt.subplot(1,5,1)
    plt.yscale("log")
    plt.scatter(c2[testing_violation2],vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
    plt.scatter(c2[testing_violation],vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
    plt.xlabel("True Constraint")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,5,2)
    plt.yscale("log")
    plt.scatter(c2[testing_violation2],vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
    plt.scatter(c2[testing_violation],vio_error[:,1],c="b",s=1,marker="x",label="beta con")
    plt.xlabel("True Constraint")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,5,3)
    plt.yscale("log")
    plt.scatter(c2[testing_violation2],vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
    plt.scatter(c2[testing_violation],vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
    plt.xlabel("True Constraint")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,5,4)
    plt.yscale("log")
    plt.scatter(c2[testing_violation2],vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
    plt.scatter(c2[testing_violation],vio_error[:,3],c="b",s=1,marker="x",label="omega con")
    plt.xlabel("True Constraint")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,5,5)
    plt.yscale("log")
    plt.scatter(c2[testing_violation2],vio_error2[:,4],c="r",s=1,marker="x",label="sigma0 no con")
    plt.scatter(c2[testing_violation],vio_error[:,4],c="b",s=1,marker="x",label="sigma0 con")
    plt.xlabel("True Constraint")
    plt.ylabel("Relative Deviation")
    plt.legend()
    
    fig = plt.figure()
    plt.suptitle('Errors Deterministic Calibration', fontsize=16)
    plt.scatter(c2,c)
    plt.plot(np.arange(0, 1.1,0.5),np.arange(0,1.1,0.5),'-r')
    plt.xlabel("True Constraint")
    plt.ylabel("Forecasted Constraint")
    
    plt.figure(figsize=(14,4))
    plt.suptitle('Errors Deterministic Calibration', fontsize=16)
    ax=plt.subplot(1,5,1)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
    plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
    plt.xlabel("constraint deviation")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,5,2)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
    plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,1],c="b",s=1,marker="x",label="beta con")
    plt.xlabel("constraint deviation")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,5,3)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
    plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
    plt.xlabel("constraint deviation")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,5,4)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
    plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,3],c="b",s=1,marker="x",label="omega con")
    plt.xlabel("constraint deviation")
    plt.ylabel("Relative Deviation")
    plt.legend()
    ax=plt.subplot(1,5,5)
    plt.yscale("log")
    plt.xscale("log")
    plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,4],c="r",s=1,marker="x",label="sigma0 no con")
    plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,4],c="b",s=1,marker="x",label="sigma0 con")
    plt.xlabel("constraint deviation")
    plt.ylabel("Relative Deviation")
    plt.legend()
    return error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2

def plotter_autoencoder(forecast,y_true_test,y_test,testing_violation,testing_violation2):
    # Example Plots
    X = strikes
    Y = maturities
    X, Y = np.meshgrid(X, Y)
    
    sample_idx = random.randint(0,len(y_test))
    
    #error plots
    mape = np.zeros(forecast.shape)
    mse  = np.zeros(forecast.shape)
    err_rel_mat  = np.zeros((Ntest,Nparameters))
    err_mat      = np.zeros((Ntest,Nparameters))
    for i in range(Ntest):
        mape[i,:,:] =  np.abs((y_true_test[i,:,:]-forecast[i,:,:])/y_true_test[i,:,:])
        mse[i,:,:]  =  np.square((y_true_test[i,:,:]-forecast[i,:,:]))
    idx = np.argsort(np.max(mape,axis=tuple([1,2])), axis=None)
    
    #bad_idx = idx[:-200]
    bad_idx = idx
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.plot_surface(X, Y, y_true_test[idx[-1],:,:], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, forecast[idx[-1],:,:] , rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('Strikes')
    ax.set_ylabel('Maturities')
    plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.plot_surface(X, Y, y_true_test[sample_idx,:,:], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, forecast[sample_idx,:,:] , rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('Strikes')
    ax.set_ylabel('Maturities')
    plt.show()
    #from matplotlib.colors import LogNorm
    plt.figure(figsize=(14,4))
    ax=plt.subplot(2,3,1)
    err1 = 100*np.mean(mape[bad_idx,:,:],axis=0)
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
    err2 = 100*np.std(mape[bad_idx,:,:],axis = 0)
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
    err3 = 100*np.max(mape[bad_idx,:,:],axis = 0)
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
    err1 = np.sqrt(np.mean(mse[bad_idx,:,:],axis=0))
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
    err2 = np.std(mse[bad_idx,:,:],axis = 0)
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
    err3 = np.max(mse[bad_idx,:,:],axis = 0)
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
    
    
    
    
    #from matplotlib.colors import LogNorm
    plt.figure(figsize=(14,4))
    plt.suptitle('Error with constraint violation', fontsize=16)
    ax=plt.subplot(2,3,1)
    bad_idx = testing_violation
    err1 = 100*np.mean(mape[bad_idx,:,:],axis=0)
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
    err2 = 100*np.std(mape[bad_idx,:,:],axis = 0)
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
    err3 = 100*np.max(mape[bad_idx,:,:],axis = 0)
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
    err1 = np.sqrt(np.mean(mse[bad_idx,:,:],axis=0))
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
    err2 = np.std(mse[bad_idx,:,:],axis = 0)
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
    err3 = np.max(mse[bad_idx,:,:],axis = 0)
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
    
    
    
    #from matplotlib.colors import LogNorm
    plt.figure(figsize=(14,4))
    plt.suptitle('Error with no constrain violation', fontsize=16)
    ax=plt.subplot(2,3,1)
    bad_idx = testing_violation2
    err1 = 100*np.mean(mape[bad_idx,:,:],axis=0)
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
    err2 = 100*np.std(mape[bad_idx,:,:],axis = 0)
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
    err3 = 100*np.max(mape[bad_idx,:,:],axis = 0)
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
    err1 = np.sqrt(np.mean(mse[bad_idx,:,:],axis=0))
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
    err2 = np.std(mse[bad_idx,:,:],axis = 0)
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
    err3 = np.max(mse[bad_idx,:,:],axis = 0)
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
    return mape,mse

### Heston Nandi Pricer
"""
Heston Nandi GARCH Option Pricing Model (2000) 
Based on the code of Dustin Zacharias, MIT, 2017
Code Available under https://github.com/SW71X/hngoption2
"""
# Trapezoidal Rule passing two vectors
def trapz(X, Y):
    n = len(X)
    sum = 0.0
    for i in range(1, n):
        sum += 0.5 * (X[i] - X[i - 1]) * (Y[i - 1] + Y[i])
    return sum

# HNC_f returns the real part of the Heston & Nandi integral with Q parameers
def HNC_f_Q(complex_phi, d_alpha, d_beta, d_gamma_star, d_omega, d_V, d_S, d_K, d_r, i_T, i_FuncNum):
    A = [x for x in range(i_T + 1)]
    B = [x for x in range(i_T + 1)]
    complex_zero = complex(0.0, 0.0)
    complex_one = complex(1.0, 0.0)
    #complex_i = complex(0.0, 1.0)
    A[i_T] = complex_zero
    B[i_T] = complex_zero
    for t in range(i_T - 1, -1, -1):
        if i_FuncNum == 1:
            A[t] = A[t + 1] + (complex_phi + complex_one) * d_r + B[t + 1] * d_omega \
                   - 0.5 * cmath.log(1.0 - 2.0 * d_alpha * B[t + 1])
            B[t] = - 0.5 * (complex_phi+complex_one)+ d_beta * B[t + 1] \
                   + (0.5 * (complex_phi+complex_one) ** 2-2*d_alpha*(d_gamma_star)*B[t+1]*(complex_phi+complex_one)\
                      +d_alpha*(d_gamma_star)**2*B[t+1] )/ (1.0 - 2.0 * d_alpha * B[t + 1])
        else:
            A[t] = A[t + 1] + (complex_phi) * d_r + B[t + 1] * d_omega \
                   - 0.5 * cmath.log(1.0 - 2.0 * d_alpha * B[t + 1])
            B[t] = - 0.5 * complex_phi + d_beta * B[t + 1] \
                   + (0.5 * (complex_phi) **2 - 2*d_alpha*(d_gamma_star)*B[t+1]*complex_phi+d_alpha*(d_gamma_star)**2*B[t+1] )\
                   / (1.0 - 2.0 * d_alpha * B[t + 1])
    if i_FuncNum == 1:
        z = (d_K ** (-complex_phi)) * (d_S ** (complex_phi + complex_one)) \
            * cmath.exp(A[0] + B[0] * d_V) / complex_phi
        return z.real
    else:
        z = (d_K ** (-complex_phi)) * (d_S ** (complex_phi)) * cmath.exp(A[0] + B[0] * d_V) / complex_phi
        return z.real
    
# Returns the Heston and Nandi option price under Q parameters
def HNC_Q(alpha, beta, gamma_star, omega, V, S, K, r, T, PutCall):
    const_pi = 4.0 * math.atan(1.0)
    High = 100#1000
    Increment = 0.25#0.05
    NumPoints = int(High / Increment)
    X, Y1, Y2 = [], [], []
    i = complex(0.0, 1.0)
    phi = complex(0.0, 0.0)
    for j in range(0, NumPoints):
        if j == 0:
            X.append(np.finfo(float).eps)
        else:
            X.append(j * Increment)
        phi = X[j] * i
        Y1.append(HNC_f_Q(phi, alpha, beta, gamma_star, omega, V, S, K, r, T, 1))
        Y2.append(HNC_f_Q(phi, alpha, beta, gamma_star, omega, V, S, K, r, T, 2))

    int1 = trapz(X, Y1)
    int2 = trapz(X, Y2)
    Call = S / 2 + math.exp(-r * T) * int1 / const_pi - K * math.exp(-r * T) * (0.5 + int2 / const_pi)
    Put = Call + K * math.exp(-r * T) - S
    if PutCall == 1:
        return Call
    else:
        return Put
    return 

def error_fun_opti(x,prediction,i):
    alpha = x[0]
    beta = x[1]
    gamma_star = x[2]
    omega = x[3] 
    h0 = x[4]
    err = 0
    for t in range(Nmaturities):
        err += ((prediction[t,i]-bsimpvola(HNC_Q(alpha, beta, gamma_star, omega, h0, 1, strikes[i], r, maturities[t], 1),1,strikes[i],maturities[t],r,'c'))\
                /prediction[t,i])**2
    return err/(Nmaturities)


def error_fun(x, prediction, i):
    return error_fun_opti(x, prediction, i)

import functools as functools

def opti_fun_data(prediction):
    def opti_fun(x):
        try:
            pool = Pool(np.max([os.cpu_count()-1,1]))
            error = np.mean(pool.map(functools.partial(error_fun, x, prediction), range(Nstrikes)))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()
        return error
    return opti_fun

import time
### timer class
class ownTimerError(Exception):

    """A custom exception used to report errors in use of Timer class"""


class ownTimer:

    def __init__(self):

        self._start_time = None


    def start(self):

        """Start a new timer"""

        if self._start_time is not None:

            raise ownTimerError(f"Timer is running. Use .stop() to stop it")


        self._start_time = time.perf_counter()


    def stop(self):

        """Stop the timer, and report the elapsed time"""

        if self._start_time is None:

            raise ownTimerError(f"Timer is not running. Use .start() to start it")


        elapsed_time = time.perf_counter() - self._start_time

        self._start_time = None

        print(f"Elapsed time: {elapsed_time:0.4f} seconds")


