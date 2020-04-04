% This code calibrates the HNG(1,1) under P with MLE. 
% calibrated on every wednessday in 2010-2018
% Windowlength 2520 days
% Parameters are scaled to [0,1]
% interest rates are calcalated for each year and used as fixed input
clc; close all; clearvars;

%% data
datatable       = readtable('SP500_220320.csv');
data            = [datenum(datatable.Date),year(datatable.Date),datatable.Close,[0;log(datatable.Close(2:end))-log(datatable.Close(1:end-1))]];
win_len         = 2520; %around 10years
years           = (data(:,2)==2010) | (data(:,2)==2011) | (data(:,2)==2012) | (data(:,2)==2013) | (data(:,2)==2014) | (data(:,2)==2015) | (data(:,2)==2016) | (data(:,2)==2017) | (data(:,2)==2018);
wednessdays     = (weekday(data(:,1))==4);
doi             = years & wednessdays; %days of interest
index           = find(doi);
shortdata       = data(doi,:);


%% optimization

% Setup and Inital Values
omega           = 1.8e-9;
alpha           = 1.5e-6;
beta            = 0.63;
gamma           = 250;
lambda          = 2.4;
sigma0          = (alpha+omega)/(1-beta-alpha*gamma.^2);
Init            = [omega,alpha,beta,gamma,lambda,sigma0];
sc_fac          = magnitude(Init);
Init_scale      = Init./sc_fac;
scaler          = sc_fac(1,:);  
lb_h0           = [1e-12,0,0,-1000,-100,1e-12];
ub_h0           = [1,1,100,2000,100,1];
A               = [];
b               = [];
Aeq             = [];
beq             = [];
% Initialisation
opt_ll          = NaN*ones(length(index),1);
params_mle_weekly=NaN*ones(length(index),6);
params_mle_weekly_original=NaN*ones(length(index),6);
hist_vola       = NaN*ones(length(index),1);
useYield = 0;
%path                = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
path                = '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
if useYield
    path_r       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
else
    path_r       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
    end
load(path_r);
for i=1:length(index)
    tmp = shortdata(i,2)-2009; %  year 
    display(datatable.Date(index(i)));
    logret = data(index(i)-win_len:index(i)-1,4);
    hist_vola(i) = sqrt(252)*std(logret);
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
    
    
    
    f_min_raw = @(par, scaler) ll_hng_n_h0(par.*scaler,logret,r);
    gs = GlobalSearch('XTolerance',1e-9,'StartPointsToRun','bounds-ineqs','Display','final');
    


    % Check two different initial values for better results.
    % the previous optimal value and the original one
    if i~=1 
        x0      = [params;Init_scale];
        fmin_   = zeros(2,1);
        xmin_   = zeros(2,6);
        for j=1:2
            if j
                scaler  = scale_tmp;
            else
                scaler  = sc_fac; 
            end
            f_min = @(params) f_min_raw(params,scaler);
            nonlincon_fun = @(params) nonlincon_scale_v2(params,scaler);
            problem = createOptimProblem('fmincon','x0',x0(j,:),...
                'objective',f_min,'lb',lb_h0./scaler,'ub',ub_h0./scaler,'nonlcon',nonlincon_fun);
            [xmin_(j,:),fmin_(j)] = run(gs,problem);
        end
        [fmin,idx] = min(fmin_);
        xmin = xmin_(idx,:);
        if idx
            scaler = scale_tmp;    
        else 
            scaler = sc_fac;
        end
        
    else
        f_min = @(params) f_min_raw(params,scaler);
        nonlincon_fun = @(params) nonlincon_scale_v2(params,scaler);
        gs = GlobalSearch('XTolerance',1e-9,...
            'StartPointsToRun','bounds-ineqs','NumTrialPoints',2e3);
        problem = createOptimProblem('fmincon','x0',Init_scale,...
                'objective',f_min,'lb',lb_h0./scaler,'ub',ub_h0./scaler,'nonlcon',nonlincon_fun);
        [xmin,fmin] = run(gs,problem);    
        
    end    
    params                          = xmin;
    params_original                 = xmin.*scaler;
    scale_tmp                       = magnitude(params_original);
    opt_ll(i)                       = fmin;
    params_mle_weekly(i,:)          = params;
    params_mle_weekly_original(i,:) = params_original;
end
params_Q_mle_weekly = [params_mle_weekly_original(:,1:3),params_mle_weekly_original(:,4)+params_mle_weekly_original(:,5)+0.5];
sig2_0 = params_mle_weekly_original(:,6);
save('weekly_10to18_mle_opt.mat','sig2_0','hist_vola','params_Q_mle_weekly')