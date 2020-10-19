% This code calibrates the HNG(1,1) under P with MLE. 
% calibrated on every Wednesday in 2010-2018
% Windowlength 2520 days
% Parameters are scaled to [0,1]
% interest rates are calcalated for each year and used as fixed input
clc; close all; clearvars;

%% data
datatable       = readtable('SP500_220320.csv');
data            = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
win_end         = 0; % around 10years
win_start       = 252;
win_start_hist  = 2520 + 252;
years           = (data(:,2)==2010) | (data(:,2)==2011) | (data(:,2)==2012) | (data(:,2)==2013) | (data(:,2)==2014) | (data(:,2)==2015) | (data(:,2)==2016) | (data(:,2)==2017) | (data(:,2)==2018);
wednesdays     = (weekday(data(:,1))==4);
doi             = years & wednesdays; %days of interest
index           = find(doi);
shortdata       = data(doi,:);


useYield = 0;
path                = 'C:/GIT/HenrikAlexJP/Data/Datasets';
load('weekly_10to18_mle_opt_h0est_check_rng_unCond.mat');
if useYield
    path_r       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
else
    path_r       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
    end
load(path_r);
sigma2_last = zeros(length(index), 1);
sigma2_upd_0 = zeros(length(index), 1);
likValQ = zeros(length(index), 1);
for i=1:length(index)
    tmp = shortdata(i,2)-2009; %  year 
    display(datatable.Date(index(i)));
    logret_hist = data(index(i)-win_start_hist + 1:index(i)-win_end,4);
    hist_vola(i) = sqrt(252)*std(logret_hist);

    logret = data(index(i)-win_start + 1:index(i)-win_end,4);
    omega = params_Q_mle_weekly(i, 1);
    alpha = params_Q_mle_weekly(i, 2);
    beta = params_Q_mle_weekly(i, 3);
    gamma_star = params_Q_mle_weekly(i, 4);
    sigma2_upd_0(i) = (alpha+omega)/(1-beta-alpha*gamma_star.^2);
    % compute interest rates for the weekly options
    if useYield
        r = SP500_date_prices_returns_realizedvariance_interestRates(8, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1));
        if isempty(r)
            r = SP500_date_prices_returns_realizedvariance_interestRates(8, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1)-1);
        end
        if all(isnan(r))
            r = SP500_date_prices_returns_realizedvariance_interestRates(8, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1)-1);
        end
    else
        r = SP500_date_prices_returns_realizedvariance_interestRates(9, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) ==shortdata(i,1));
        if isempty(r)
            r = SP500_date_prices_returns_realizedvariance_interestRates(9, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1)-1);
        end
        if all(isnan(r))
            r = SP500_date_prices_returns_realizedvariance_interestRates(9, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1)-1);
        end
    end
    r=max(r,0)/252;    

    [likValQ(i), sigma2_vals] = ll_hng_Q_n(params_Q_mle_weekly(i, :), logret, r, sigma2_upd_0(i));
    sigma2_last(i) = sigma2_vals(end);
    
end

save('weekly_10to18_mle_opt_h0est_check_rng_unCond.mat','sig2_0','hist_vola', 'opt_ll','likValQ','sigma2_last',...
     'sigma2_upd_0', 'hist_vola', '-append');
