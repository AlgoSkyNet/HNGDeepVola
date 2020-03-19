%Describtion "test_data_with_errors_alex.mat"
clc; close all;clearvars;
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
vio_error               = mape errors of prediction_invtrafo
vio_error2              = mape errors of prediction_invtrafo

mse_autoencoder         = mse of the autoencoder on test set           
mape_autoencoder        = mape of autoencoder on test set


Ntest/Nval/Ntrain   
Nparameters/Nstrikes
/Nmaturities            = size of corresponding variable
S0                      = vector of initial underlying / unnecessary for now   
%}

%visualisation
r = 0.005;
data_vec = [combvec(strikes,maturities);S0*ones(1,Nmaturities*Nstrikes)]';
delete(gcp('nocreate'))
pool_  = parpool('local',4);
j = 0;
num = 1:Ntest;
num_vio = num(testing_violation);
params_vio = prediction_invtrafo(testing_violation,:);
% for i = 1:length(vio_error)
%     if mod(i,50)==0
%         disp(i)
%     end
%     j = num_vio(i);
%     params  = prediction_invtrafo(j,:);
%     imp_vola_params(i,:) = blsimpv_vec(data_vec,r,price_Q_clear([params(4),params(1),params(2),params(3)],data_vec,r/252,params(5)));
%     prices_params(i,:) = price_Q_clear([params(4),params(1),params(2),params(3)],data_vec,r/252,params(5));
% end
% save('imp_vola.mat','imp_vola_params','prices_params')
load("imp_vola.mat")
figure
y = strikes;
x = maturities;
[X,Y] = meshgrid(x,y);
for i=1:8
subplot(2,4,i)
surf(x,y,reshape(imp_vola_params(randi(length(vio_error),1,1),:),9,7));hold on;
set(gca,'Zscale','log')
end
figure
for i=1:8
subplot(2,4,i)
surf(x,y,reshape(prices_params(randi(length(vio_error),1,1),:),9,7));hold on;
set(gca,'Zscale','log')
end

figure
subplot(2,5,1)
xx = params_vio;
histogram(xx(:,1),'Normalization','probability')
subplot(2,5,2)
histogram(xx(:,2),'Normalization','probability')
subplot(2,5,3)
histogram(xx(:,3),'Normalization','probability')
subplot(2,5,4)
histogram(xx(:,4),'Normalization','probability')
subplot(2,5,5)
histogram(xx(:,5),'Normalization','probability')
subplot(2,5,6)
boxplot(xx(:,1))
subplot(2,5,7)
boxplot(xx(:,2))
subplot(2,5,8)
boxplot(xx(:,3))
subplot(2,5,9)
boxplot(xx(:,4))
subplot(2,5,10)
boxplot(xx(:,5))
figure
subplot(1,2,1)
histogram(c(testing_violation),'Normalization','probability')
subplot(1,2,2)
boxplot(c(testing_violation))

