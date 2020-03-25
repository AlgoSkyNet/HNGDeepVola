%HNG-Optimization under Q 
%Options: path = '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
clc; 
clearvars; 
close all;


%parpool()
path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
stock_ind           =  'SP500';
year                =  2015;
path_               =  strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);
% load Interest rates
load('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/InterestRates/interestRates2015.mat');

load('weekly_2015_mle.mat');

% if use realized volatility data then load the corresponding data
useRealVola = 1;
if useRealVola
    path_vola       =  strcat(path, '/', stock_ind, '/', 'SP500_date_prices_returns_realizedvariance_090320.mat');
    load(path_vola);
end

bound                   = [100, 100];
formatIn                = 'dd-mmm-yyyy';

% start from the first Wednesday of 2015 and finish with the last Wednesday of 2015

DateString_start        = '07-January-2015';
DateString_end          = '30-December-2015';
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
Dates                   = date_start:7:date_end;

% initialize with the data from MLE estimation for each week
Init                    = params_Q_mle_weekly;
% bounds for maturity, moneyness, volumes, interest rates
Type                    = 'call';
MinimumVolume           = 100;
MinimumOpenInterest     = 100;
IfCleanNans             = 1;
TimeToMaturityInterval  = [8, 250];
MoneynessInterval       = [0.9, 1.1];

[OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptions(Dates, Type, ...
    TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans,...
    TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume, ...
    OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
    TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);

weeksprices             = week(datetime([OptionsStruct.date], 'ConvertFrom', 'datenum'));
idx                     = zeros(length(weeksprices), max(weeksprices));

j = 1;
for i = min(weeksprices):max(weeksprices)
    idx(:, j) = (weeksprices == i)';
    j = j + 1;
end

data = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega];
save('generaldata2015.mat', 'data', 'DatesClean', 'OptionsStruct', 'OptFeatures', 'idx');
%% Optiimization

% Initialization     
r                =   interestRates(4,2)/252;
sc_fac           =   magnitude(Init);
Init_scale_mat   =   Init./sc_fac;
lb_mat           =   [1e-12, 0, 0, -500]./sc_fac;
ub_mat           =   [1, 1, 10, 1000]./sc_fac;
opt_params_raw   =   zeros(max(weeksprices), 4);
opt_params_clean =   zeros(max(weeksprices), 4);
values           =   cell(1,max(weeksprices));
Init_scale       =   Init_scale_mat(1, :);
scaler           =   sc_fac(1, :);  

        
% weekly optimization
j = 1;
for i = min(weeksprices):max(weeksprices)
    if useRealVola
        sig2_0(i) = SP500_date_prices_returns_realizedvariance_090320(4, ...
            SP500_date_prices_returns_realizedvariance_090320(1,:) == Dates(j));
    end
    data_week = data(:, logical(idx(:,j))')';
    j = j + 1;
    if isempty(data_week)
        continue
    end
    %lower parameter bounds, scaled
    lb = lb_mat(i, :);
    %upper parameter bounds, scaled
    ub = ub_mat(i, :); 

    % Goal function
    
    % RMSE
    %f_min = @(params) sqrt(mean((price_Q(params.*scaler,data_week,r,sig2_0(i))'-data_week(:,1)).^2));
    
    % MSE
    f_min_raw = @(params, scaler) (mean((price_Q(params.*scaler, data_week, r, sig2_0(i))' - data_week(:, 1)).^2));
    
    % MRAE/MAPE
    %f_min_raw = @(params,scaler) mean(abs(price_Q(params.*scaler,data_week,r,sig2_0(i))'-data_week(:,1))./data_week(:,1));
    
    %  Interior Point
    opt = optimoptions('fmincon', 'Display', 'iter',...
        'Algorithm', 'interior-point', 'MaxIterations', 1000,...
        'MaxFunctionEvaluations', 1500, 'TolFun',1e-3, 'TolX', 1e-3);
    
    % SQP
    % opt = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxIterations',50,'MaxFunctionEvaluations',300,'FunctionTolerance',1e-4);
    
    % Starting value check / semi globalization
    if i ~= min(weeksprices)
        x1      = Init_scale_mat(i, :);
        scaler  = sc_fac(i, :); 
        f1      = f_min_raw(x1, scaler);
        x2      = opt_params_raw(i - 1, :);
        scaler  = scale_tmp;
        f2      = f_min_raw(x2, scaler);
        if f1 < f2
            Init_scale = x1;
            scaler = sc_fac(i, :);
    
        else 
            Init_scale = x2;
            scaler = scale_tmp;
        end
            
    end
    f_min = @(params) f_min_raw(params, scaler); 
    
    % run optimization
    nonlincon_fun = @(params) nonlincon_scale_v2(params, scaler);
    opt_params_raw(i, :) = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    % store the results

    opt_params_clean(i,:) = opt_params_raw(i, :).*scaler;
    struc           =   struct();
    struc.Price     =   data_week(:, 1)';
    struc.hngPrice  =   price_Q(opt_params_clean(i,:), data_week, r, sig2_0(i)) ;
    struc.numOptions =  length(data_week(:, 1));

    % compute interest rates for the weekly options
    r_cur = zeros(length(data_week), 1);
    for k = 1:length(data_week)
        r_cur(k) = spline(interestRates(:,1), interestRates(:,2), data_week(k, 2));
    end
    struc.blsPrice  =   blsprice(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, hist_vola(i), 0)';
    struc.blsimpv   =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, data_week(:, 1));
    struc.blsimpvhn =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, struc.hngPrice');
    struc.epsilon   =   (struc.Price - struc.hngPrice) ./ data_week(:,5);
    s_epsilon2      =   mean(struc.epsilon(:).^2);
    epsilon2_norm   =   sum((struc.epsilon(:).^2)) / s_epsilon2;
    struc.optionsLik =  (-.5 * (struc.numOptions * (log(2 * pi) + log(s_epsilon2)) + epsilon2_normalized));
    struc.meanPrice =   mean(data_week(:, 1));
    struc.hngparams =   opt_params_clean(i, :);
    struc.countneg  =   sum(struc.hngPrice <= 0);
    struc.matr      =   [struc.Price; struc.hngPrice; struc.blsPrice];
    struc.maxAbsEr  =   max(abs(struc.hngPrice - struc.Price));
    struc.IVRMSE    =   sqrt(mean(100 * (struc.blsimpv - struc.blsimpvhn).^2));
    struc.MAPE      =   mean(abs(struc.hngPrice - struc.Price)./struc.Price);
    struc.MaxAPE    =   max(abs(struc.hngPrice - struc.Price)./struc.Price);
    struc.RMSE      =   sqrt(mean((struc.hngPrice - struc.Price).^2));
    struc.RMSEbls   =   sqrt(mean((struc.blsPrice - struc.Price).^2));
    struc.scale     =   scaler;
    scale_tmp       =   scaler;
    values{i}       =   struc;
    
end 

save('params_Options_2015_h0asRealVola_MSE.mat','values');