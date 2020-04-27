% This code calibrates the HNG(1,1) under P with MLE. 
% calibrated on every Wednesday in 2010-2018
% Windowlength 2520 days
% Parameters are scaled to [0,1]
% interest rates are calcalated for each year and used as fixed input
clc; close all; clearvars;
%% data
datatable       = readtable('SP500_220320.csv');
data            = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
win_len         = 2520; % around 10years
years           = (data(:,2)==2010) | (data(:,2)==2011) | (data(:,2)==2012) | (data(:,2)==2013) | (data(:,2)==2014) | (data(:,2)==2015) | (data(:,2)==2016) | (data(:,2)==2017) | (data(:,2)==2018);
wednesdays     = (weekday(data(:,1))==4);
doi             = years & wednesdays; %days of interest
index           = find(doi);
shortdata       = data(doi,:);

ifEstimater = 1;
%% optimization
ifEstimateh0 = 1;
% Setup and Inital Values
if ifEstimateh0
    num_params  = 6;
else
    num_params  = 5;
end
if ifEstimater
    num_params = num_params + 1;
end

omega           = 1.8e-9;
alpha           = 1.5e-6;
beta            = 0.63;
gamma           = 250;
lambda          = 2.4;
sigma0          = (alpha+omega)/(1-beta-alpha*gamma.^2);
if ifEstimateh0
    Init        = [omega,alpha,beta,gamma,lambda,sigma0];
    lb_h0       = [1e-12,0,0,-1000,-100,1e-12];
    ub_h0       = [1,1,100,2000,100,1];
else
    Init        = [omega,alpha,beta,gamma,lambda];
    lb_h0       = [1e-12,0,0,-1000,-100];
    ub_h0       = [1,1,100,2000,100];
end
sc_fac          = magnitude(Init);
        Init_scale      = Init./sc_fac;
        scaler          = sc_fac(1,:); 
A               = [];
b               = [];
Aeq             = [];
beq             = [];
% Initialisation
opt_ll                      = NaN*ones(length(index),1);
params_mle_weekly           = NaN*ones(length(index),num_params);
params_mle_weekly_original  = NaN*ones(length(index),num_params);
hist_vola                   = NaN*ones(length(index),1);
sigma2_last                 = NaN*ones(length(index),1);
useYield = 0;

%path                = '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
path                = 'C:/GIT/HenrikAlexJP/Data/Datasets';
if useYield
    path_r       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
else
    path_r       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
    end
load(path_r);
tic;
for i=1:length(index)
    tmp = shortdata(i,2)-2009; %  year 
    display(datatable.Date(index(i)));
    toc
    logret = data(index(i)-win_len:index(i)-1,4);
    hist_vola(i) = sqrt(252)*std(logret);
    % compute interest rates for the weekly options
    if useYield
         dates_oi = data(index(i)-win_len:index(i)-1,1);
        [ind1,ind2] = find(SP500_date_prices_returns_realizedvariance_interestRates(1,:) == dates_oi);
        r = SP500_date_prices_returns_realizedvariance_interestRates(8, ind2);
        r = nanmean(r);
%         r = SP500_date_prices_returns_realizedvariance_interestRates(8, ...
%             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1));
%         if isempty(r)
%             r = SP500_date_prices_returns_realizedvariance_interestRates(8, ...
%                 SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1)-1);
%         end
%         if all(isnan(r))
%             r = SP500_date_prices_returns_realizedvariance_interestRates(8, ...
%                 SP500_date_prices_returns_realizedvariance_interestRates(1,:) == shortdata(i,1)-1);
%         end
    else
        %dates = data(index(i)-win_len:index(i)-1,1);
        %[ig1,ig2] = find(SP500_date_prices_returns_realizedvariance_interestRates(1,:) ==dates);
        %r = SP500_date_prices_returns_realizedvariance_interestRates(9, ...
        %    ig2);
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
    if ifEstimater
        if i == 1
            Init        = [Init, r];
            lb_h0       = [lb_h0, 1e-12];
            ub_h0       = [ub_h0, 1e-5];
        else
            Init(end) = r;
        end
        sc_fac          = magnitude(Init);
        Init_scale      = Init./sc_fac;
        scaler          = sc_fac(1,:); 
    end
    if ifEstimateh0
        if ~ifEstimater
            f_min_raw = @(par, scaler) ll_hng_n_h0(par.*scaler,logret,r);
        else
            f_min_raw = @(par, scaler) ll_hng_n_h0r(par.*scaler,logret);
        end
    elseif (~ifEstimateh0 && ifEstimater)
         f_min_raw = @(par, scaler) ll_hng_n_r(par.*scaler,logret,sigma0);
    else
        f_min_raw = @(par, scaler) ll_hng_n(par.*scaler,logret,r,sigma0);
    end
    gs = GlobalSearch('XTolerance',1e-9,'FunctionTolerance', 1e-9,...
            'StartPointsToRun','bounds-ineqs','NumTrialPoints',2e3,'Display','final');
    


    % Check two different initial values for better results.
    % the previous optimal value and the original one
    if i~=1 
        x0      = [params;Init_scale];
        fmin_   = zeros(2,1);
        xmin_   = zeros(2,num_params);
        for j=1:2
            if j
                scaler  = scale_tmp;
            else
                scaler  = sc_fac; 
            end
            f_min = @(params) f_min_raw(params,scaler);
            nonlincon_fun = @(params) nonlincon_scale_v2(params,scaler);
            rng('default');
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
        rng('default');
        problem = createOptimProblem('fmincon','x0',Init_scale,...
                'objective',f_min,'lb',lb_h0./scaler,'ub',ub_h0./scaler,'nonlcon',nonlincon_fun);
       [xmin,fmin] = run(gs,problem);  
%         ms = MultiStart('UseParallel', true, 'XTolerance',1e-9,...
%             'FunctionTolerance', 1e-9);    
%         options_multistart = optimoptions(@fmincon, 'Algorithm', 'interior-point', ...
%                     'TolFun', 1e-9, 'TolX', 1e-9, 'Display', 'iter', 'UseParallel', 'Always', ...
%                     'MaxIter', 200, 'MaxFunEvals', 10000);
%         problem = createOptimProblem('fmincon','x0',Init_scale,...
%                 'objective',f_min,'lb',lb_h0./scaler,'ub',ub_h0./scaler,'nonlcon',nonlincon_fun,...
%                 'options', options_multistart);
%          stpoints = RandomStartPointSet('NumStartPoints',1e5);
%         [xmin,fmin] = run(ms,problem,4);    
        
    end    
    params                          = xmin;
    params_original                 = xmin.*scaler;
    scale_tmp                       = magnitude(params_original);
    opt_ll(i)                       = -fmin;
    params_mle_weekly(i,:)          = params;
    params_mle_weekly_original(i,:) = params_original;
%     if ifEstimateh0
%         if ~ifEstimater
%             [likVal, sigma2_last(i)] = ll_hng_n_h0(params_original,logret,r);
%         else
%             [likVal, sigma2_last(i)] = ll_hng_n_h0r(params_original,logret);
%         end
%     else
%         [likVal, sigma2_last(i)] = ll_hng_n(params_original,logret,r,sigma0);
%     end
    
end

params_P_mle_weekly = [params_mle_weekly_original(:,1:3),params_mle_weekly_original(:,4),params_mle_weekly_original(:,5)];
params_Q_mle_weekly = [params_mle_weekly_original(:,1:3),params_mle_weekly_original(:,4)+params_mle_weekly_original(:,5)+0.5];
r = params_mle_weekly_original(:,7);
if ifEstimateh0
    sig2_0 = params_mle_weekly_original(:,num_params);
else
    sig2_0 = sigma0*ones(length(index),1);
end

save('weekly_10to18_mle_opt_h0est_rEst_rng.mat','sig2_0','hist_vola', 'opt_ll','sigma2_last',...
    'params_Q_mle_weekly','params_P_mle_weekly','r')
save('weekly_10to18_mle_opt_h0est_rEst_rng_allResSaved.mat')