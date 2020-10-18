clc; close all; clearvars;
serverNames     = {'MSPablo', 'Sargas', 'local'};

%% options for the simulation type
useYield = 1;
useAverage = 1; %ATTENTION! can be used only together with useYield = 1;
useServer = 3;
ifEstimateh0 = 1;
ifEstimater = 0;

%% data
datatable       = readtable('SP500_220320.csv');
data            = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
win_end         = 0;
win_start       = 252;
win_start_hist  = 2520;
years           = (data(:,2)==2010) | (data(:,2)==2011) | (data(:,2)==2012) | (data(:,2)==2013) | (data(:,2)==2014) | (data(:,2)==2015) | (data(:,2)==2016) | (data(:,2)==2017) | (data(:,2)==2018);
wednesdays     = (weekday(data(:,1))==4);
doi             = years & wednesdays; %days of interest
index           = find(doi);
shortdata       = data(doi,:);

switch useServer
    case 1
        pathPrefix = 'C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/';
    case 2
        pathPrefix = 'C:/GIT/HenrikAlexJP/';
    otherwise
        pathPrefix = '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/';
end


fileFolderNamePart2 = 'rWeek for Update/';
if useYield
    path_r       =  strcat(pathPrefix, 'Data/Datasets/InterestRates/SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
    if useAverage
        pathType_r   = 'Yields/r average/';
        fileFolderNamePart2 = 'rAv for Update/';
        filePrefixType = '_rAvYield';
    else
        pathType_r   = 'Yields/r Week/';
        filePrefixType = '_rWeekYield';
    end
    ind_r = 8;
else
    path_r       =  strcat(pathPrefix, 'Data/Datasets/InterestRates/SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
    pathType_r   = 'Tbills/r Week/';
    if useAverage
        disp('No r average for Tbills');
        exit;
    end
    filePrefixType = '_rWeekTbill';
    ind_r = 9;
end

if ifEstimateh0
    pathType_r = strcat(pathType_r, 'estimated h0P/');
    fileFolderNamePart1 = 'Results with estimated h0P ';
    filePrefixh0 = '_h0est';
else
    pathType_r = strcat(pathType_r, 'not estimated h0P/');
    fileFolderNamePart1 = 'Results with not estimated h0P ';
    filePrefixh0 = '_noh0est';
end

fileName = strcat('weekly_10to18_mleP', filePrefixh0, filePrefixType, '_UncQUpdate.mat');
totalFileNameWithPath = (strcat(pathPrefix,'Code/calibration checks/Calibration MLE P/paper version/data for tables/', pathType_r, fileFolderNamePart1, [' ' fileFolderNamePart2], fileName));
load(totalFileNameWithPath);


load(path_r);
sigma2_last = zeros(length(index), 1);
sigma2_upd_0 = zeros(length(index), 1);
likValQ = zeros(length(index), 1);
for i=1:length(index)
    display(datatable.Date(index(i)));
    % historical vola is computed using the last 10 years to make
    % comparabel to other strategies
    logret_hist = data(index(i) - win_start_hist:index(i) - 1,4);
    hist_vola(i) = sqrt(252) * std(logret_hist);
    % the function gives h_{t+1} hence in order to stop on a given
    % Wednesday with the corresponding h we use the sample up to Tuesday
    logret = data(index(i) - win_start:index(i),4);
    omega = params_Q_mle_weekly(i, 1);
    alpha = params_Q_mle_weekly(i, 2);
    beta = params_Q_mle_weekly(i, 3);
    gamma_star = params_Q_mle_weekly(i, 4);
    sigma2_upd_0(i) = (alpha+omega)/(1-beta-alpha*gamma_star.^2);
    % compute interest rates for the weekly options
    if useAverage
        dates_oi = data(index(i) - win_start + 1:index(i) - win_end,1);
        [ind1, ind2] = find(SP500_date_prices_returns_realizedvariance_interestRates(1,:) == dates_oi);
        r = SP500_date_prices_returns_realizedvariance_interestRates(ind_r, ind2);
        r = nanmean(r);
    else
        r = SP500_date_prices_returns_realizedvariance_interestRates(ind_r, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1));
        if isempty(r)
            r = SP500_date_prices_returns_realizedvariance_interestRates(ind_r, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1)-1);
        end
        if all(isnan(r))
            r = SP500_date_prices_returns_realizedvariance_interestRates(ind_r, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1)-1);
        end
    end
    r = max(r,0)/252;
    [likValQ(i), sigma2_vals] = ll_hng_Q_n_paper(params_Q_mle_weekly(i, :), logret, r, sigma2_upd_0(i));
    sigma2_last(i) = sigma2_vals(end);
    rValues_for_UpdatePeriod(i) = r;
end

save(totalFileNameWithPath, 'hist_vola','likValQ', 'sigma2_last',...
    'sigma2_upd_0', 'hist_vola', 'rValues_for_UpdatePeriod', '-append');
