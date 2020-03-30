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
global_idx          = 0; %indicator if globalisation algorithm should be used.
algorithm           = "interior-point";% "sqp"
goal                =  "OptLL"; % "MSE";   "MAPE";  ,"OptLL";
path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);
load(strcat('C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/Calibration MLE/','weekly_',num2str(year),'_mle_opt.mat'));

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
load(strcat('C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/Calibration MLE/','weekly_',num2str(year),'_mle_opt.mat'));
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

idxj  = 1:length(unique(weeksprices));



data = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega; OptionsStruct.implied_volatility];
% save('generaldata2015.mat', 'data', 'DatesClean', 'OptionsStruct', 'OptFeatures', 'idx');
%% Optimization

% Initialization     
sc_fac           =   magnitude(Init);
Init_scale_mat   =   Init./sc_fac;
lb_mat           =   [1e-12, 0, 0, -1500];
ub_mat           =   [1, 1, 10, 1500];
opt_params_raw   =   zeros(max(weeksprices), 4);
opt_params_clean =   zeros(max(weeksprices), 4);
values           =   cell(1,max(weeksprices));
sig2_0           =   zeros(1,max(weeksprices));

%values in first iteration:
Init_scale       =   Init_scale_mat(min(weeksprices), :);
scaler           =   sc_fac(min(weeksprices), :);  

       
% weekly optimization
j = 1;
for i = unique(weeksprices)
    disp(strcat("Optimization (",goal ,") of week ",num2str(i)," in ",num2str(year),"."))
    if useRealVola
        if isempty(SP500_date_prices_returns_realizedvariance_interestRates(4,...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)))
            sig2_0(i) = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-1);
        else
            sig2_0(i) = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j));
        end
    end
    data_week = data(:,(weeksprices == i))';
    if isempty(data_week)
        continue
    end

    
    struc               =  struct();
    struc.numOptions    =  length(data_week(:, 1));
    % compute interest rates for the weekly options
    if useYield
        interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:8, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j));
        if isempty(interestRates)
            interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:8, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-1);
        end
        if all(isnan(interestRates))
            interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:8, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-1);
        end
    else
        interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:9, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j));
        if isempty(interestRates)
            interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:9, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-1);
        end
        if all(isnan(interestRates))
            interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:9, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-1);
        end
    end
    for k = 1:length(interestRates)
        if interestRates(k)<0
            interestRates(k)=0;
        end
    end
    j = j + 1;
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
    struc.yields        =   interestRates;
    
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

    init_f = 1;
    % Starting value check / semi globalization
    if i ~= min(weeksprices)
        x1      = Init_scale_mat(i, :);
        scaler  = sc_fac(i, :); 
        f1      = f_min_raw(x1, scaler);
        x2      = opt_params_raw(i - 1, :);
        scaler  = scale_tmp;
        f2      = f_min_raw(x2, scaler);
        x3      = opt_params_raw(min(weeksprices), :);
        scaler  = scaler_firstweek; 
        f3      = f_min_raw(x3, scaler);
        [init_f,min_i]    = min([f1,f2,f3]);
        if min_i == 1
            Init_scale = x1;
            scaler = sc_fac(i, :);
        elseif min_i == 2
            Init_scale = x2;
            scaler = scale_tmp;
        elseif min_i ==3
            Init_scale = x3;
            scaler = scaler_firstweek;
        end
    end
    f_min = @(params) f_min_raw(params, scaler); 
    
    % run optimization
    nonlincon_fun = @(params) nonlincon_scale_v2(params, scaler);
    %lower parameter bounds, scaled
    lb = lb_mat./scaler;
    %upper parameter bounds, scaled
    ub = ub_mat./scaler; 
    
    opt = optimoptions('fmincon', ...
            'Display', 'iter',...
            'Algorithm', algorithm,...
            'MaxIterations', 300,...
            'MaxFunctionEvaluations',1500, ...
            'TolFun', 1e-6,...
            'TolX', 1e-6,...
            'TypicalX',Init(i,:)/scaler);
    struc.optispecs = struct();
    struc.optispecs.optiopt = opt;
    struc.optispecs.scaleproblem = 0;
    if magnitude(init_f)>100 
        opt.ScaleProblem = 'obj-and-constr' ; %scale to goalfunc to [0,1]
        struc.scaleproblem = 1;
    end
    if global_idx
        gs = GlobalSearch('XTolerance',1e-6,...
                          'FunctionTolerance',1e-4,...
                          'StartPointsToRun','bounds-ineqs',...
                          'Display','iter',...
                          'NumTrialPoints', 1000,...
                          'NumStageOnePoints',200);
        problem = createOptimProblem('fmincon','x0',Init_scale,...
                    'objective',f_min,'lb',lb,'ub',ub,'nonlcon',nonlincon_fun,'options',opt);
        [xxval,fval,exitflag,gsresults,gsvec] = run(gs,problem);
        struc.optispecs.gsopt   = gs;
        struc.optispecs.gsresults = gsresults;
        struc.optispecs.localminima = gsvec;
    else
        [xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    end
    if (global_idx==0) && (fval>2*f_val_firstweek) && (i ~= min(weeksprices))
        gs = GlobalSearch('XTolerance',1e-6,...
                          'FunctionTolerance',1e-4,...
                          'StartPointsToRun','bounds-ineqs',...
                          'Display','iter',...
                          'NumTrialPoints', 1000,...
                          'NumStageOnePoints',200);
        f_min = @(params) f_min_raw(params, scaler); 
        nonlincon_fun = @(params) nonlincon_scale_v2(params, scaler);
        x_gloopt =xxval*scaler;
        scaler = magnitude(x_gloopt);
        Init_scale = x_gloopt/scaler; 
        lb = lb_mat./scaler;
        ub = ub_mat./scaler;
        problem = createOptimProblem('fmincon','x0',Init_scale,...
                    'objective',f_min,'lb',lb,'ub',ub,'nonlcon',nonlincon_fun,'options',opt);
        [xxval,fval,exitflag,gsresults,gsvec] = run(gs,problem);
        struc.optispecs.gsopt   = gs;
        struc.optispecs.gsresults = gsresults;
        struc.optispecs.localminima = gsvec;
    end
       
    opt_params_raw(i, :) = xxval;
    struc.optispecs.flag = exitflag;
    struc.optispecs.goalval = fval;
    opt_params_clean(i, :) = opt_params_raw(i, :).*scaler;   
    scale_tmp           =   magnitude(opt_params_clean(i, :));
    if i == min(weeksprices)
        scaler_firstweek= scale_tmp;
        f_val_firstweek = fval;
    end
    struc.optispecs.scale         =   scale_tmp;
        
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
    values{i}           =   struc;    
end 
if strcmp(algorithm,"interior-point") %for file naming purposes
    algorithm = "interiorpoint";
end
save(strcat('params_Options_',num2str(year),'_h0asRealVola_',goal,'_',algorithm,'_',txt,'.mat'),'values');

%for specific weeks
%save(strcat('params_Options_',num2str(year),'week2and4','_h0asRealVola_',goal,'_',algorithm,'_',txt,'.mat'),'values');