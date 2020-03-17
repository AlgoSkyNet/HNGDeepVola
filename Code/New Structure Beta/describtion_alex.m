%Describtion "test_data_with_errors_alex.mat"
load("test_data_with_errors_alex.mat")
%{
X_test                  = true parameters of the test set
X_test_trafo2           = minmax trafo of true parameters of the test set (see masterthesis chapter 4)
y_test                  = true vola surfaces test set vectorized
y_test_re / y_true_test = true vola surfaces test set
maturities / strikes    = parameters for the grid

forecast                = NN2(NN1(y_true_test) autoencoder forecast 
prediction              = NN1(X_test_trafo2) surface forecast
prediction_calibration  = NN2(y_test_re)  parameter forecast of NN2
prediction_invtrafo     = parameter forecast of NN2 scaled to original scale
                          (inv of minmax scaling)


c                       = constraint of prediction_invtrafo
c2                      = constraint of test set
testing_violation       = boolean  shows whether constrain is violated or not for prediction inv_trafo / >=1  
testing_violation2      = NOT(testing_violation) / negative boolean of testing_violation 
vio_error               = c(testing_violation )
vio_error2              = c(testing_violation2) 

mse_autoencoder         = mse of the autoencoder on test set           
mape_autoencoder        = mape of autoencoder on test set


Ntest/Nval/Ntrain   
Nparameters/Nstrikes
/Nmaturities            = size of corresponding variable
S0                      = vector of initial underlying / unnecessary for now   
%}

