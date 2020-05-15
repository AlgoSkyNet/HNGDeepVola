% this file restructures the cummulated output of the MLE calibration into
% yearly files.
clc; clearvars;close all;
load('weekly_10to18_mle_opt_noh0Est_norEst_rng.mat')
DateString_start        = '01-January-2010';
DateString_end          = '31-December-2018';  
formatIn                = 'dd-mmm-yyyy';
datatable       = readtable('SP500_220320.csv');
data            = [datenum(datatable.Date),year(datatable.Date)];
idx =1;
for y = 2010:2018
    i = y-2009; 
    years           = (data(:,2)==y); 
    wednessdays     = (weekday(data(:,1))==4);
    doi             = years & wednessdays; %days of interest
    weeks           = int64(week(datatable.Date(doi)));
    num_            = sum(doi);
    
    sig_tmp         = zeros(53,1);
    vola_tmp        = zeros(53,1);
    params_tmp      = zeros(53,4);
    sig0_tmp        = zeros(53,1);
    sig0_unc_tmp    = zeros(53,1);
    disp([idx,num_]);
        params_P                  = params_P_mle_weekly(idx:idx+num_-1,:);

    sig                     = sigma2_last(idx:idx+num_-1);
    sig_old                 = sig2_0(idx:idx+num_-1);
    vola                    = hist_vola(idx:idx+num_-1);
    params                  = params_Q_mle_weekly(idx:idx+num_-1,:);
    sig_unc                 = (params(:,1)+params(:,2))./(1-params(:,2).*params(:,4).^2-params(:,3));
        logLik = opt_ll(idx:idx+num_-1);
    
    idx = idx+num_;
    logLikVals(weeks,:)       = logLik;
    sig_tmp(weeks,:)        = sig;
    sig0_tmp(weeks,:)       = sig_old;
    sig0_unc_tmp(weeks,:)   = sig_unc;
    vola_tmp(weeks,:)       = vola;
    params_tmp(weeks,:)     = params;
    params_tmp_P(weeks,:)     = params_P;
    name                    = strcat('weekly_',num2str(i+2009),'_mle_opt_noh0est.mat');
    save(name,'sig_tmp','vola_tmp','params_tmp','sig0_tmp','sig0_unc_tmp','logLikVals','params_tmp_P')
end