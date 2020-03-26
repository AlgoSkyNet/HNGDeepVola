%HNG-Optimization under Q 
%Options: path = '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
clc; 
clearvars; 
close all;


%parpool()
path                = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
stock_ind           = 'SP500';
year                = 2010;
useYield            = 0; %uses tbils now
useRealVola         = 1; %alwas use realized vola
algorithm           = "InteriorPoint";% "SQP"
goal                = "MSE"; % "MSE";   "MAPE";  ,"OptLL";
path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);
load(strcat('weekly_',num2str(year),'_mle_opt.mat'));

% load Interest rates
% load the corresponding data
if useYield
    path_vola       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
    txt = "useYield";
else
    path_vola       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
    txt = "noYield";
end
load(path_vola);

% if use realized volatility data then load the corresponding data


bound                   = [100, 100];
formatIn                = 'dd-mmm-yyyy';

% start from the first Wednesday of 2015 and finish with the last Wednesday of 2015

DateString_start        = strcat('01-January-',num2str(year));
DateString_end          = strcat('31-December-',num2str(year));
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
wednessdays             = (weekday(date_start:date_end)==4);
Dates                   = date_start:date_end;
Dates                   = Dates(wednessdays);

% initialize with the data from MLE estimation for each week
Init                    = params_tmp;
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

data = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega; OptionsStruct.implied_volatility];
% save('generaldata2015.mat', 'data', 'DatesClean', 'OptionsStruct', 'OptFeatures', 'idx');
%% Optimization

% Initialization     
sc_fac           =   magnitude(Init);
Init_scale_mat   =   Init./sc_fac;
lb_mat           =   [1e-12, 0, 0, -500]./sc_fac;
ub_mat           =   [1, 1, 10, 1000]./sc_fac;
opt_params_raw   =   zeros(max(weeksprices), 4);
opt_params_clean =   zeros(max(weeksprices), 4);
values           =   cell(1,max(weeksprices));
%values in first iteration:
Init_scale       =   Init_scale_mat(min(weeksprices), :);
scaler           =   sc_fac(min(weeksprices), :);  

       
% weekly optimization
j = 1;
for i = unique(weeksprices)%min(weeksprices):max(weeksprices)
    disp(strcat("Optimization of week ",num2str(i)," in year ",num2str(year),"."))
    if useRealVola
        sig2_0(i) = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j));
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
    
    struc               =   struct();
    struc.numOptions    =  length(data_week(:, 1));
    % compute interest rates for the weekly options
    if useYield
        interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:8, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j));
    else
            interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:9, ...
        SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j));
    end
    
    r_cur = zeros(length(data_week), 1);
    if useYield
        for k = 1:length(data_week)
            if data_week(k, 2) < 21
                r_cur(k) = interestRates(1);
            else
                r_cur(k) = interp1([21,63,126,252]./252, interestRates, data_week(k, 2)./252);
            end
            %r_cur(k) = spline(interestRates(:,1), interestRates(:,2), data_week(k, 2));
        end
    else
        for k = 1:length(data_week)
            if data_week(k, 2) < 21 && ~isnan(interestRates(1))
                r_cur(k) = interestRates(1);
            else
                notNaN = ~isnan(interestRates);
                daylengths = [21, 42, 13*5, 126, 252]./252;
                r_cur(k) = interp1(daylengths(notNaN), interestRates(notNaN), data_week(k, 2)./252);
                if isnan(r_cur(k))
                    b=0;
                end
            end
            %r_cur(k) = spline(interestRates(:,1), interestRates(:,2), data_week(k, 2));
        end
    end
    struc.blsPrice      =   blsprice(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, vola_tmp(i), 0)';
    struc.blsimpv       =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, data_week(:, 1));
    struc.Price         =   data_week(:, 1)';
    struc.sig20         =   sig2_0(i);
    struc.yields        =   interestRates(notNaN);
    
    %% Goal function

    % MSE
    if strcmp(goal,"MSE")
        f_min_raw = @(params, scaler) (mean((price_Q(params.*scaler, data_week, r_cur./252, sig2_0(i))' - data_week(:, 1)).^2));
    % MRAE/MAPE
    elseif strcmp(goal,"MAPE")
        f_min_raw = @(params,scaler) mean(abs(price_Q(params.*scaler, data_week, r_cur./252, sig2_0(i))'-data_week(:, 1))./data_week(:, 1));
    % Option Likelyhood
    elseif strcmp(goal,"OptLL")
        f_min_raw = @(params,scaler) (0.5 * struc.numOptions * (log(2*pi) + 1 + log(mean(((price_Q(params.*scaler, data_week, r_cur./252, sig2_0(i))'-data_week(:, 1))./data_week(:, 5)).^2))));
    % WE DO NOT USE THIS FOR GOAL FUNCTION
    % RMSE
    %elseif strcmp(goal,"RMSE")
    %    f_min_raw = @(params, scaler) sqrt(mean((price_Q(params.*scaler,data_week,r,sig2_0(i))'-data_week(:,1)).^2));
    % IV RMSE
    %elseif strcmp(goal,"IVRMSE")
    %   f_min_raw = @(params, scaler) sqrt(mean(100 * (struc.blsimpv - blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, price_Q(params.*scaler, data_week, r_cur./252, sig2_0(i))')).^2));
    end
    
    %% Algorithm 
    %  Interior Point
    if strcmp(algorithm,"InteriorPoint")
        opt = optimoptions('fmincon', 'Display', 'iter',...
        'Algorithm', 'interior-point', 'MaxIterations', 1000,...
        'MaxFunctionEvaluations',2000, 'TolFun', 1e-6, 'TolX', 1e-6);
    % WE DO NOT USE THIS AS OPTIMIZATION ALGORITHM FUNCTION
    % SQP
    %elseif strcmp(algorithm,"SQP")
    %    opt = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxIterations',50,'MaxFunctionEvaluations',300,'FunctionTolerance',1e-4);
    end
    
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
    opt_params_clean(i, :) = opt_params_raw(i, :).*scaler;
    
    
    struc.hngPrice      =   price_Q(opt_params_clean(i,:), data_week, r_cur./252, sig2_0(i)) ;
    struc.blsimpvhng    =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, struc.hngPrice');
    struc.epsilonhng    =   (struc.Price - struc.hngPrice) ./ data_week(:,5)';
    struc.epsilonbls    =   (struc.Price - struc.blsPrice) ./ data_week(:,5)';
    s_epsilon2hng       =   mean(struc.epsilonhng(:).^2);
    s_epsilon2bls       =   mean(struc.epsilonbls(:).^2);
    struc.optionsLikhng =   -.5 * struc.numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1);
    struc.optionsLikbls =   -.5 * struc.numOptions * (log(2 * pi) + log(s_epsilon2bls) + 1);
    struc.meanPrice     =   mean(data_week(:, 1));
    struc.hngparams     =   opt_params_clean(i, :);
    struc.countneg      =   sum(struc.hngPrice <= 0);
    struc.matr          =   [struc.Price; struc.hngPrice; struc.blsPrice];
    struc.maxAbsEr      =   max(abs(struc.hngPrice - struc.Price));
    struc.IVRMSE        =   sqrt(mean(100 * (struc.blsimpv - struc.blsimpvhng).^2));
    struc.MAPE          =   mean(abs(struc.hngPrice - struc.Price)./struc.Price);
    struc.MaxAPE        =   max(abs(struc.hngPrice - struc.Price)./struc.Price);
    struc.MSE           =   mean((struc.hngPrice - struc.Price).^2);
    struc.RMSE          =   sqrt(struc.MSE);
    struc.RMSEbls       =   sqrt(mean((struc.blsPrice - struc.Price).^2));
    struc.scale         =   scaler;
    scale_tmp           =   scaler;
    values{i}           =   struc;    
end 
save(strcat('params_Options_',num2str(year),'_h0asRealVola_',goal,'_',algorithm,'_',txt,'.mat'),'values');

%for specific weeks
%save(strcat('params_Options_',num2str(year),'week2and4','_h0asRealVola_',goal,'_',algorithm,'_',txt,'.mat'),'values');