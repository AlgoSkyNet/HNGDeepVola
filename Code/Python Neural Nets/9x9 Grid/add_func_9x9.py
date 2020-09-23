# In[Preambel]:
import numpy as np
from tensorflow.keras import backend as K
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

# import custom functions #scaling tools
from config_9x9 import ytransform, yinversetransform,myscale,myinverse
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#from py_vollib.black_scholes.implied_volatility import implied_volatility as bsimpvola
#from py_vollib.black_scholes.implied_volatility import black_scholes
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


# In[Custom Errors]:
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
def miss_count(y_true, y_pred):
    traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
    traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
    traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
    constraint = traf_a*K.square(traf_g)+traf_b
    return K.mean(K.greater(constraint,1))

def mse_constraint(param):
    def rel_mse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.mean(K.square(y_pred - y_true)) +param*K.mean(K.greater(constraint,1))
    return rel_mse_constraint

def log_constraint(param,p2=30):
    def log_mse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.mean(K.square(y_pred - y_true)) +param*K.mean(1/(1+K.exp(-p2*(constraint-1))))
    return log_mse_constraint
def l2rel_log_constraint(param,p2=30):
    def log_rmse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.sqrt(K.mean(K.square((y_pred - y_true)/y_true))) +param*K.mean(1/(1+K.exp(-p2*(constraint-1))))
    return log_rmse_constraint

def mape_constraint(param,p2=30):
    def r_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.mean(K.abs((y_true - y_pred) / y_true)) +param*K.mean(1/(1+K.exp(-p2*(constraint-1))))
    return r_constraint

def log_constraint_noscale(param,p2=30):
    def log_mse_constraint_noscale(y_true, y_pred):
            constraint = y_pred[:,0]*K.square(y_pred[:,2])+y_pred[:,1]
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.mean(K.square(y_pred - y_true)) +param*K.mean(1/(1+K.exp(-p2*(constraint-1))))
    return log_mse_constraint_noscale

def constraint_violation(x):
    return np.sum(x[:,0]*x[:,2]**2+x[:,1]>=1)/x.shape[0],x[:,0]*x[:,2]**2+x[:,1]>=1,x[:,0]*x[:,2]**2+x[:,1]

# In[ErrorPlots]:

# error pricer
def pricing_plotter(prediction,y_test,vega_test):     
    err_rel_mat  = np.zeros(prediction.shape)
    err_mat      = np.zeros(prediction.shape)
    err_optll    = np.zeros(prediction.shape)
    err_iv_approx= np.zeros(prediction.shape)
    for i in range(y_test.shape[0]):
        err_rel_mat[i,:,:]  =  np.abs((y_test[i,:,:]-prediction[i,:,:])/y_test[i,:,:])
        err_mat[i,:,:]      =  np.square((y_test[i,:,:]-prediction[i,:,:]))
        err_optll[i,:,:]    = np.log(np.square((y_test[i,:,:]-prediction[i,:,:])/vega_test[i,:,:])) 
        err_iv_approx[i,:,:]= np.square((y_test[i,:,:]-prediction[i,:,:])/vega_test[i,:,:]) 
    idx = np.argsort(np.max(err_rel_mat,axis=tuple([1,2])), axis=None)
    
    #bad_idx = idx[:-200]
    bad_idx = idx
    #from matplotlib.colors import LogNorm
    fig = plt.figure()
    plt.suptitle('Errors Neural Net Pricing', fontsize=16)
    
    
    
    ax=plt.subplot(4,3,1)
    err1 = 100*np.mean(err_rel_mat[bad_idx,:,:],axis=0)
    plt.title("Average relative error",fontsize=10,y=1.04)
    plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.colorbar(format=mtick.PercentFormatter())
    ax.axes.get_xaxis().set_visible(False)
    #ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    #ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    #plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(4,3,2)
    err2 = 100*np.std(err_rel_mat[bad_idx,:,:],axis = 0)
    plt.title("Std relative error",fontsize=10,y=1.04)
    plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
    plt.colorbar(format=mtick.PercentFormatter())
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    #ax.set_xticklabels(strikes)
    #ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    #ax.set_yticklabels(maturities)
    #plt.xlabel("Strike",fontsize=10,labelpad=5)
    #plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(4,3,3)
    err3 = 100*np.max(err_rel_mat[bad_idx,:,:],axis = 0)
    plt.title("Maximum relative error",fontsize=10,y=1.04)
    plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
    plt.colorbar(format=mtick.PercentFormatter())
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    #ax.set_xticklabels(strikes)
    #ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    #ax.set_yticklabels(maturities)
    #plt.xlabel("Strike",fontsize=10,labelpad=5)
    #plt.ylabel("Maturity",fontsize=10,labelpad=5)
    
    
    
    ax=plt.subplot(4,3,4)
    err1 = np.mean(err_mat[bad_idx,:,:],axis=0)
    plt.title("MSE",fontsize=10,y=1.04)
    plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
    plt.colorbar()#format=mtick.PercentFormatter())
    ax.axes.get_xaxis().set_visible(False)
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    #ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    #plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(4,3,5)
    err2 = np.std(err_mat[bad_idx,:,:],axis = 0)
    plt.title("Std MSE",fontsize=10,y=1.04)
    plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
    plt.colorbar()#format=mtick.PercentFormatter())
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    #ax.set_xticklabels(strikes)
    #ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    #ax.set_yticklabels(maturities)
    #plt.xlabel("Strike",fontsize=10,labelpad=5)
    #plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(4,3,6)
    err3 = np.max(err_mat[bad_idx,:,:],axis = 0)
    plt.title("Maximum MSE",fontsize=10,y=1.04)
    plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
    plt.colorbar()#format=mtick.PercentFormatter())
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    #ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    #ax.set_xticklabels(strikes)
    #ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    #ax.set_yticklabels(maturities)
    #plt.xlabel("Strike",fontsize=10,labelpad=5)
    #plt.ylabel("Maturity",fontsize=10,labelpad=5)
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    ax=plt.subplot(4,3,7)
    err1 = np.mean(err_optll[bad_idx,:,:],axis=0)
    plt.title("OptLL",fontsize=10,y=1.04)
    plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
    plt.colorbar()#format=mtick.PercentFormatter())
    ax.axes.get_xaxis().set_visible(False)
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    #ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    #plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(4,3,8)
    err2 = np.std(err_optll[bad_idx,:,:],axis = 0)
    plt.title("Std OPtll",fontsize=10,y=1.04)
    plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
    plt.colorbar()#format=mtick.PercentFormatter())
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    #ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    #ax.set_xticklabels(strikes)
    #ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    #ax.set_yticklabels(maturities)
    #plt.xlabel("Strike",fontsize=10,labelpad=5)
    #plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(4,3,9)
    err3 = np.max(err_optll[bad_idx,:,:],axis = 0)
    plt.title("Maximum Optll",fontsize=10,y=1.04)
    plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
    plt.colorbar()#format=mtick.PercentFormatter())
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    #ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    #ax.set_xticklabels(strikes)
    #ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    #ax.set_yticklabels(maturities)
    #plt.xlabel("Strike",fontsize=10,labelpad=5)
    #plt.ylabel("Maturity",fontsize=10,labelpad=5)
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    
    ax=plt.subplot(4,3,10)
    err1 = np.sqrt(np.mean(err_iv_approx[bad_idx,:,:],axis=0))
    plt.title("IVRMSE Approx",fontsize=10,y=1.04)
    plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
    plt.colorbar()#format=mtick.PercentFormatter())
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(4,3,11)
    err2 = np.std(err_iv_approx[bad_idx,:,:],axis = 0)
    plt.title("Std IVRMSE",fontsize=10,y=1.04)
    plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
    plt.colorbar()#format=mtick.PercentFormatter())
    ax.axes.get_yaxis().set_visible(False)
    
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    ax.set_xticklabels(strikes)
    #ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    #ax.set_yticklabels(maturities)
    plt.xlabel("Strike",fontsize=10,labelpad=5)
    #plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(4,3,12)
    err3 = np.max(err_iv_approx[bad_idx,:,:],axis = 0)
    plt.title("Maximum IVRMSE",fontsize=10,y=1.04)
    plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
    plt.colorbar()#format=mtick.PercentFormatter())
    ax.axes.get_yaxis().set_visible(False)
    
    ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    ax.set_xticklabels(strikes)
    #ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    #ax.set_yticklabels(maturities)
    plt.xlabel("Strike",fontsize=10,labelpad=5)
    #plt.ylabel("Maturity",fontsize=10,labelpad=5)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.tight_layout()
    
    #for ax in fig.get_axes():
    #    ax.label_outer()
    plt.show()
    
    return err_rel_mat,err_mat,err_optll,err_iv_approx,idx,bad_idx

#errror calibration
def calibration_plotter(prediction_calibration,X_test_trafo2,X_test,extra_plots = 0):
    prediction_invtrafo= np.array([myinverse(x) for x in prediction_calibration])
    prediction_std = np.std(prediction_calibration,axis=0)
    Ndata = prediction_invtrafo.shape[0]
    error = np.zeros((Ndata,Nparameters))

    for i in range(Ndata):
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
    bad_scenarios = prediction_invtrafo[testing_violation,:]
    testing_violation2 = (c<1)
    vio_error = error[testing_violation,:]
    vio_error2 = error[testing_violation2,:]
    
    # BOXPLOTS
    if vio_error.size != 0:  
        plt.figure(figsize=(14,4))
        plt.suptitle("Boxploit of Rel Errors Neural Net Calibration per Parameter", fontsize=16)
    
        ax=plt.subplot(1,3,1)
        plt.boxplot(error)
        #plt.yscale("log")
        plt.xticks([1, 2, 3,4,5], ['a','b','g*',"w","h0"])
        plt.ylabel("Errors")
        ax=plt.subplot(1,3,2)
        plt.boxplot(vio_error)
        #plt.yscale("log")
        plt.xticks([1, 2, 3,4,5], ['a','b','g*',"w","h0"])
        plt.ylabel("Errors parameter violation")
        ax=plt.subplot(1,3,3)
        plt.boxplot(vio_error2)
        #plt.yscale("log")
        plt.xticks([1, 2, 3,4,5], ['a','b','g*',"w","h0"])
        plt.ylabel("Errors no parameter violation")
        plt.show()
        plt.show()
    #HISTOGRAM
    plt.figure(figsize=(14,4))
    plt.suptitle('Rel Errors Neural Net Calibration per Parameter', fontsize=16)
    legendlist = ['a','b','g*',"w",'h0']
    for i in range(Nparameters):
        ax=plt.subplot(2,Nparameters,i+1)
        plt.xscale("log")
        plt.hist(100*error[:,i],bins=np.logspace(np.log10(0.001),np.log10(10000), 200))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xlabel("relative deviation")
        ax=plt.subplot(2,5,i+6)
        plt.yscale("log")
        plt.boxplot(100*error[:,i])
        plt.xticks([1],legendlist[i])
        if i==0:
            plt.ylabel("Relative Deviation")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()

    
    
    print("violation error mean in %:",100*np.mean(vio_error,axis=0))
    print("no violation error mean in %:",100*np.mean(vio_error2,axis=0))
    print("violation error median in %:",100*np.median(vio_error,axis=0))
    print("no violation error median in %:",100*np.median(vio_error2,axis=0))
    print("error mean in %:",100*err1)
    print("error median in %:",100*err2)
    
    
    if extra_plots:
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
        
    return error,err1,err2,vio_error,vio_error2,c,c2,testing_violation,testing_violation2,bad_scenarios

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
    """
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
    plt.show()"""
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


def vola_plotter(prediction,y_test):     
    err_rel_mat  = np.zeros(prediction.shape)
    err_mat      = np.zeros(prediction.shape)
    for i in range(y_test.shape[0]):
        err_rel_mat[i,:,:]  =  np.abs((y_test[i,:,:]-prediction[i,:,:])/y_test[i,:,:])
        err_mat[i,:,:]      =  np.square((y_test[i,:,:]-prediction[i,:,:]))
    idx = np.argsort(np.max(err_rel_mat,axis=tuple([1,2])), axis=None)
    
    #bad_idx = idx[:-200]
    bad_idx = idx
    #from matplotlib.colors import LogNorm
    fig = plt.figure()
    plt.suptitle('Errors Neural Net Pricing', fontsize=16)
    
    
    
    ax=plt.subplot(2,3,1)
    err1 = 100*np.mean(err_rel_mat[bad_idx,:,:],axis=0)
    plt.title("Average relative error",fontsize=10,y=1.04)
    plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.colorbar(format=mtick.PercentFormatter(),fraction=0.05, pad=0.05)
    ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(2,3,2)
    err2 = 100*np.std(err_rel_mat[bad_idx,:,:],axis = 0)
    plt.title("Std relative error",fontsize=10,y=1.04)
    plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
    plt.colorbar(format=mtick.PercentFormatter(),fraction=0.05, pad=0.05)

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(2,3,3)
    err3 = 100*np.max(err_rel_mat[bad_idx,:,:],axis = 0)
    plt.title("Maximum relative error",fontsize=10,y=1.04)
    plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
    plt.colorbar(format=mtick.PercentFormatter(),fraction=0.05, pad=0.05)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    
    
    
    ax=plt.subplot(2,3,4)
    err1 = np.mean(err_mat[bad_idx,:,:],axis=0)
    plt.title("MSE",fontsize=10,y=1.04)
    plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
    plt.colorbar(fraction=0.05, pad=0.05)#format=mtick.PercentFormatter())
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(2,3,5)
    err2 = np.std(err_mat[bad_idx,:,:],axis = 0)
    plt.title("Std MSE",fontsize=10,y=1.04)
    plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
    plt.colorbar(fraction=0.05, pad=0.05)#format=mtick.PercentFormatter())
    
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    ax=plt.subplot(2,3,6)
    err3 = np.max(err_mat[bad_idx,:,:],axis = 0)
    plt.title("Maximum MSE",fontsize=10,y=1.04)
    plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
    plt.colorbar(fraction=0.05, pad=0.05)#format=mtick.PercentFormatter())
     
    ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
    ax.set_xticklabels(strikes)
    ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
    ax.set_yticklabels(maturities)
    plt.xlabel("Strike",fontsize=10,labelpad=5)
    plt.ylabel("Maturity",fontsize=10,labelpad=5)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.tight_layout()
    
    #for ax in fig.get_axes():
    #    ax.label_outer()
    plt.show()
    
    return err_rel_mat,err_mat
