% this file restructures the cummulated output of the MLE calibration into
% yearly files.
clc; clearvars;close all;
load('weekly_10to18_mle_opt.mat')
DateString_start        = '01-January-2010';
DateString_end          = '31-December-2018';  
formatIn                = 'dd-mmm-yyyy';
datatable       = readtable('SP500_220320.csv');
data            = [datenum(datatable.Date),year(datatable.Date)];
for y = 2010:2018
    years           = (data(:,2)==y); 
    wednessdays     = (weekday(data(:,1))==4);
    doi             = years & wednessdays; %days of interest
    num_(y-2009)    = sum(doi);
    idx             = cumsum(num_);
end 
idx             = cumsum(num_);
idx = int64([0,idx]);
for i = 1:9
    disp([idx(i)+1,idx(i+1)]);
    sig_tmp                 = sig2_0(idx(i)+1:idx(i+1));
    vola_tmp                = hist_vola(idx(i)+1:idx(i+1));
    params_tmp              = params_Q_mle_weekly(idx(i)+1:idx(i+1));
    name                    = strcat('weekly_',num2str(i+2009),'_mle_opt.mat');
    save(name,'sig_tmp','vola_tmp','params_tmp')
end