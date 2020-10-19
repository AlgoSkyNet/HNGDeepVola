% plot(cumsum(IVRMSE_est),'DisplayName','IVRMSE_est');
% hold on;
% plot(cumsum(IVRMSE_P),'DisplayName','IVRMSE_P');
% plot(cumsum(IVRMSE_Q),'DisplayName','IVRMSE_Q');
% plot(cumsum(IVRMSE_RV),'DisplayName','IVRMSE_RV');
% hold off;
% 
% 
% plot(cumsum(MSE_est),'DisplayName','MSE_est');
% hold on;
% plot(cumsum(MSE_P),'DisplayName','MSE_P');
% plot(cumsum(MSE_Q),'DisplayName','MSE_Q');
% plot(cumsum(MSE_RV),'DisplayName','MSE_RV');
% hold off;
% 
% plot(cumsum(optLL_val_est),'DisplayName','optLL_val_est');
% hold on;
% plot(cumsum(optLL_val_P),'DisplayName','optLL_val_P');
% plot(cumsum(optLL_val_Q),'DisplayName','optLL_val_Q');
% plot(cumsum(optLL_val_RV),'DisplayName','optLL_val_RV');
% hold off;
clear;
load('s2010.mat');
i=1;
a(i,1) = mean(optLL_val_est)
b(i,1) = mean(optLL_val_P)
c(i,1) = mean(optLL_val_Q)
d(i,1) = mean(optLL_val_RV)

a(i,2) =  mean(MSE_est)
b(i,2) = mean(MSE_P)
c(i,2) = mean(MSE_Q)
d(i,2) = mean(MSE_RV)

a(i,3) = mean(IVRMSE_est)
b(i,3) = mean(IVRMSE_P)
c(i,3) = mean(IVRMSE_Q)
d(i,3) = mean(IVRMSE_RV)

load('s2011.mat');
i=2;
a(i,1) = mean(optLL_val_est)
b(i,1) = mean(optLL_val_P)
c(i,1) = mean(optLL_val_Q)
d(i,1) = mean(optLL_val_RV)

a(i,2) =  mean(MSE_est)
b(i,2) = mean(MSE_P)
c(i,2) = mean(MSE_Q)
d(i,2) = mean(MSE_RV)

a(i,3) = mean(IVRMSE_est)
b(i,3) = mean(IVRMSE_P)
c(i,3) = mean(IVRMSE_Q)
d(i,3) = mean(IVRMSE_RV)
load('s2012.mat');
i=3;
a(i,1) = mean(optLL_val_est)
b(i,1) = mean(optLL_val_P)
c(i,1) = mean(optLL_val_Q)
d(i,1) = mean(optLL_val_RV)

a(i,2) =  mean(MSE_est)
b(i,2) = mean(MSE_P)
c(i,2) = mean(MSE_Q)
d(i,2) = mean(MSE_RV)

a(i,3) = mean(IVRMSE_est)
b(i,3) = mean(IVRMSE_P)
c(i,3) = mean(IVRMSE_Q)
d(i,3) = mean(IVRMSE_RV)


load('s2013.mat');
i=4;
a(i,1) = mean(optLL_val_est)
b(i,1) = mean(optLL_val_P)
c(i,1) = mean(optLL_val_Q)
d(i,1) = mean(optLL_val_RV)

a(i,2) =  mean(MSE_est)
b(i,2) = mean(MSE_P)
c(i,2) = mean(MSE_Q)
d(i,2) = mean(MSE_RV)

a(i,3) = mean(IVRMSE_est)
b(i,3) = mean(IVRMSE_P)
c(i,3) = mean(IVRMSE_Q)
d(i,3) = mean(IVRMSE_RV)


load('s2014.mat');
i=5;
a(i,1) = mean(optLL_val_est)
b(i,1) = mean(optLL_val_P)
c(i,1) = mean(optLL_val_Q)
d(i,1) = mean(optLL_val_RV)

a(i,2) =  mean(MSE_est)
b(i,2) = mean(MSE_P)
c(i,2) = mean(MSE_Q)
d(i,2) = mean(MSE_RV)

a(i,3) = mean(IVRMSE_est)
b(i,3) = mean(IVRMSE_P)
c(i,3) = mean(IVRMSE_Q)
d(i,3) = mean(IVRMSE_RV)


load('s2015.mat');
i=6;
a(i,1) = mean(optLL_val_est)
b(i,1) = mean(optLL_val_P)
c(i,1) = mean(optLL_val_Q)
d(i,1) = mean(optLL_val_RV)

a(i,2) =  mean(MSE_est)
b(i,2) = mean(MSE_P)
c(i,2) = mean(MSE_Q)
d(i,2) = mean(MSE_RV)

a(i,3) = mean(IVRMSE_est)
b(i,3) = mean(IVRMSE_P)
c(i,3) = mean(IVRMSE_Q)
d(i,3) = mean(IVRMSE_RV)


load('s2016.mat');
i=7;
a(i,1) = mean(optLL_val_est)
b(i,1) = mean(optLL_val_P)
c(i,1) = mean(optLL_val_Q)
d(i,1) = mean(optLL_val_RV)

a(i,2) =  mean(MSE_est)
b(i,2) = mean(MSE_P)
c(i,2) = mean(MSE_Q)
d(i,2) = mean(MSE_RV)

a(i,3) = mean(IVRMSE_est)
b(i,3) = mean(IVRMSE_P)
c(i,3) = mean(IVRMSE_Q)
d(i,3) = mean(IVRMSE_RV)


load('s2017.mat');
i=8;
a(i,1) = mean(optLL_val_est)
b(i,1) = mean(optLL_val_P)
c(i,1) = mean(optLL_val_Q)
d(i,1) = mean(optLL_val_RV)

a(i,2) =  mean(MSE_est)
b(i,2) = mean(MSE_P)
c(i,2) = mean(MSE_Q)
d(i,2) = mean(MSE_RV)

a(i,3) = mean(IVRMSE_est)
b(i,3) = mean(IVRMSE_P)
c(i,3) = mean(IVRMSE_Q)
d(i,3) = mean(IVRMSE_RV)


load('s2018.mat');
i=9;
a(i,1) = mean(optLL_val_est)
b(i,1) = mean(optLL_val_P)
c(i,1) = mean(optLL_val_Q)
d(i,1) = mean(optLL_val_RV)

a(i,2) =  mean(MSE_est)
b(i,2) = mean(MSE_P)
c(i,2) = mean(MSE_Q)
d(i,2) = mean(MSE_RV)

a(i,3) = mean(IVRMSE_est)
b(i,3) = mean(IVRMSE_P)
c(i,3) = mean(IVRMSE_Q)
d(i,3) = mean(IVRMSE_RV)







