%HNG-Optimization under Q
%Options: path = '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
clc;
clearvars;
close all;
warning('on')

%parpool()
%path                = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
%path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
path                =  'C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';
year                = 2010;
useYield            = 1; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 1; % use last h_t from MLE under P as h0
num_voladays        = 2; % if real vola, give the number of historic volas used (6 corresponds to today plus 5 days = 1week);
algorithm           = 'interior-point';% 'sqp'
goal                =  'OptLL'; % 'MSE';   'MAPE';  ,'OptLL';
path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);

% load Interest rates
% load the corresponding data
if useYield
    path_vola       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
    txt = 'useYield';
else
    path_vola       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
    txt = 'noYield';
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
%load(strcat('C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/Calibration MLE/','weekly_',num2str(year),'_mle_opt.mat'));
%load(strcat('C:/Users/TEMP/Documents/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/MLE_P estimation results/','weekly_',num2str(year),'_mle_opt.mat'));
%load(strcat('C:/Users/lyudmila/Documents/GitHub/HenrikAlexJP/Code/calibration checks/Calibration MLE P/correct Likelihood/Yields/Results with estimated h0P rAv/','weekly_',num2str(year),'_mle_opt_h0est_rAv.mat'));
load(strcat('C:/Users/lyudmila/Documents/GitHub/HenrikAlexJP/Code/calibration checks/Calibration MLE P/correct Likelihood/Yields/Results with estimated h0P rAv/','weekly_',num2str(year),'_mle_opt_h0est_rAv.mat'));

if useRealVola || useMLEPh0
    num_params = 4;
else
    num_params = 5;
end

Init = params_tmp;
if ~(useRealVola || useMLEPh0)
    Init = [params_tmp,sig_tmp];
end

% bounds for maturity, moneyness, volumes, interest rates
Type                    = 'call';
MinimumVolume           = 100;
MinimumOpenInterest     = 100;
IfCleanNans             = 1;
TimeToMaturityInterval  = [8, 250];
MoneynessInterval       = [0.9, 1.1];

[OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptionsFilt(Dates, Type, ...
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
ub_mat           =   [1, 1, 1, 1500];

if ~(useRealVola || useMLEPh0)
    lb_mat = [lb_mat, 1e-12];
    ub_mat = [ub_mat, 1];
end
opt_params_raw   =   zeros(max(weeksprices), num_params);
opt_params_clean =   zeros(max(weeksprices), num_params);
values           =   cell(1,max(weeksprices));
sig2_0           =   zeros(1,max(weeksprices));

%values in first iteration:
Init_scale       =   Init_scale_mat(min(weeksprices), :);
scaler           =   sc_fac(min(weeksprices), :);


%% weekly optimization
j = 1;
good_i =[];
bad_i =[];
for i = unique(weeksprices)
    if useRealVola
        disp(strcat('Optimization (',goal ,') of week ',num2str(i),' in ',num2str(year),'. h_0 is not calibrated.'))
        vola_vec = zeros(1,num_voladays);
        vola_cell = {};
        %         vola_cell{1} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
        %             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j));
        vola_cell{1} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-1);
        vola_cell{2} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-2);
        %         vola_cell{4} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
        %             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-3);
        %         vola_cell{5} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
        %             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-4);
        %         vola_cell{6} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
        %             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-5);
        %         vola_cell{7} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
        %             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-6);
        %         vola_cell{8} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
        %             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-7);
        for vola_idx = 1:num_voladays
            if ~isempty(vola_cell{vola_idx})
                vola_vec(vola_idx) = vola_cell{vola_idx};
            end
        end
        [~,vola_idx] =max(vola_vec>0);
        sig2_0(i) = vola_vec(vola_idx);
        disp(strcat('Optimization (',goal ,') of week ',num2str(i),' in ',num2str(year),'. h_0 = realized vola'))
    elseif useMLEPh0
        disp(strcat('Optimization (',goal ,') of week ',num2str(i),' in ',num2str(year),'. h_0 = h_t from MLE under P.'))
        sig2_0(i) = sig_tmp(i);
    else
        disp(strcat('Optimization (',goal ,') of week ',num2str(i),' in ',num2str(year),'. h_0 will be calibrated.'))
    end
    data_week = data(:,(weeksprices == i))';
    if isempty(data_week)
        disp(strcat('no data for week ',num2str(i),' in ',num2str(year),'!'))
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
    struc.Price         =   data_week(:, 1)';
    struc.yields        =   interestRates;
    struc.blsPrice      =   blsprice(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, vola_tmp(i), 0)';
    struc.blsimpv       =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, data_week(:, 1));
    indNaN = find(isnan(struc.blsimpv));
    struc.num_NaN_implVols = length(indNaN);
    struc.blsimpv(indNaN) = data_week(indNaN, 6);
    struc.blsvega = blsvega(data_week(:, 4),  data_week(:, 3), r_cur(:), data_week(:, 2)/252, struc.blsimpv(:));
    
    %% Goal function
    if useRealVola || useMLEPh0
        % MSE
        if strcmp(goal,'MSE')
            f_min_raw = @(params, scaler,h0) (mean((price_Q(params.*scaler, data_week, r_cur./252, h0)' - data_week(:, 1)).^2));
            % MRAE/MAPE
        elseif strcmp(goal,'MAPE')
            f_min_raw = @(params,scaler,h0) mean(abs(price_Q(params.*scaler, data_week, r_cur./252, h0)'-data_week(:, 1))./data_week(:, 1));
            % Option Likelyhood
        elseif strcmp(goal,'OptLL')
            %f_min_raw = @(params,scaler,h0) 0.5+1/1000*(0.5 * struc.numOptions * (log(2*pi) + 1 + log(mean(((price_Q(params.*scaler, data_week, r_cur./252, h0)'-data_week(:, 1))./struc.blsvega).^2))));
            f_min_raw = @(params,scaler,h0) ((log(mean(((price_Q(params.*scaler, data_week, r_cur./252, h0)'-data_week(:, 1))./struc.blsvega).^2))));
            % WE DO NOT USE THIS FOR GOAL FUNCTION
            % RMSE
            %elseif strcmp(goal,'RMSE')
            %    f_min_raw = @(params, scaler,h0) sqrt(mean((price_Q(params.*scaler,data_week,r,h0)'-data_week(:,1)).^2));
            % IV RMSE
            %elseif strcmp(goal,'IVRMSE')
            %   f_min_raw = @(params, scaler,h0) sqrt(mean(100 * (struc.blsimpv - blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, price_Q(params.*scaler, data_week, r_cur./252, h0)')).^2));
        end
    else
        if strcmp(goal,'MSE')
            f_min_raw = @(params, scaler) (mean((price_Q_h0(params.*scaler, data_week, r_cur./252)' - data_week(:, 1)).^2));
            % MRAE/MAPE
        elseif strcmp(goal,'MAPE')
            f_min_raw = @(params,scaler) mean(abs(price_Q_h0(params.*scaler, data_week, r_cur./252)'-data_week(:, 1))./data_week(:, 1));
            % Option Likelyhood
        elseif strcmp(goal,'OptLL')
            f_min_raw = @(params,scaler) ((log(mean(((price_Q_h0(params.*scaler, data_week, r_cur./252)'-data_week(:, 1))./struc.blsvega).^2))));
            % WE DO NOT USE THIS FOR GOAL FUNCTION
            % RMSE
            %elseif strcmp(goal,'RMSE')
            %    f_min_raw = @(params, scaler) sqrt(mean((price_Q_h0(params.*scaler,data_week,r)'-data_week(:,1)).^2));
            % IV RMSE
            %elseif strcmp(goal,'IVRMSE')
            %   f_min_raw = @(params, scaler) sqrt(mean(100 * (struc.blsimpv - blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, price_Q_h0(params.*scaler, data_week, r_cur./252)')).^2));
        end
    end
    
    %% Algorithm
    
    % Starting value check
    if i ~= min(weeksprices)
        if useRealVola || useMLEPh0
            %MLE
            x1      = Init_scale_mat(i, :);
            scaler  = sc_fac(i, :);
            f1      = f_min_raw(x1, scaler,sig2_0(i));
            % previous week
            scaler  = scale_tmp;
            x2      = opt_params_clean(i - 1, :)./scaler;
            f2      = f_min_raw(x2, scaler,sig2_0(i));
            %first weeks results
            scaler  = scaler_firstweek;
            x3      = opt_params_clean(min(weeksprices), :)./scaler;
            f3      = f_min_raw(x3, scaler,sig2_0(i));
            %best weeks results
            scaler = best_scaler;
            x4 = best_x./scaler;
            f4 = f_min_raw(x4, scaler,sig2_0(i));
        else
            %MLE
            x1      = Init_scale_mat(i, :);
            scaler  = sc_fac(i, :);
            f1      = f_min_raw(x1,scaler);
            % previous week
            scaler  = scale_tmp;
            x2      = opt_params_clean(i - 1, :)./scaler;
            f2      = f_min_raw(x2,scaler);
            %first weeks results
            scaler  = scaler_firstweek;
            x3      = opt_params_clean(min(weeksprices), :)./scaler;
            f3      = f_min_raw(x3, scaler);
            %best weeks results
            scaler = best_scaler;
            x4 = best_x./scaler;
            f4 = f_min_raw(x4, scaler);
        end
        
        [~,min_i]    = min([f1,f2,f3,f4]);
        if min_i == 1
            Init_scale = x1;
            scaler = sc_fac(i, :);
            disp(strcat('Initial value used ''MLE parameters''.'));
        elseif min_i == 2
            Init_scale = x2;
            scaler = scale_tmp;
            disp(strcat('Initial value used ''previous week''.'));
        elseif min_i ==3
            Init_scale = x3;
            scaler = scaler_firstweek;
            disp(strcat('Initial value used ''first week''.'));
        elseif min_i ==4
            Init_scale = x4;
            scaler = best_scaler;
            disp(strcat('Initial value used ''best week''.'));
        end
    else
        disp(strcat('Initial value used ''MLE parameters''.'));
    end
    
    % fun2opti,scaled
    if useRealVola || useMLEPh0
        f_min = @(params) f_min_raw(params(1:num_params), scaler, sig2_0(i));
    else
        f_min = @(params) f_min_raw(params, scaler);
    end
    % constraint,scaled
    nonlincon_fun = @(params) nonlincon_scale_v2(params, scaler);
    %parameter bounds, scaled
    lb = lb_mat./scaler;
    ub = ub_mat./scaler;
    %optimization specs
    if useRealVola || useMLEPh0
        opt = optimoptions('fmincon', ...
            'Display', 'iter',...
            'Algorithm', algorithm,...
            'MaxIterations', 30000,...
            'MaxFunctionEvaluations',20000, ...
            'TolFun', 1e-6,...
            'TolX', 1e-9,...
            'TypicalX', Init(i,:)./scaler);
    else
        opt = optimoptions('fmincon', ...
            'Display', 'iter',...
            'Algorithm', algorithm,...
            'MaxIterations', 4000,...
            'MaxFunctionEvaluations',2500, ...
            'TolFun', 1e-15,...
            'TolX', 1e-15,...
            'TypicalX',Init(i,:)./scaler);
    end
    
    struc.optispecs = struct();
    struc.optispecs.optiopt = opt;
    
    %local optimization
    % [xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    rng('default');
    gs = GlobalSearch('XTolerance',1e-15,'FunctionTolerance', 1e-15,...
        'StartPointsToRun','bounds-ineqs','NumTrialPoints',5e2,  'Display', 'iter');
    problem = createOptimProblem('fmincon','x0',Init_scale,...
        'objective',f_min,'lb',lb,'ub',ub,'nonlcon',nonlincon_fun);
    [xxval,fval,exitflag] = run(gs,problem);
    %
    %            ms = MultiStart('XTolerance',1e-9,'FunctionTolerance', 1e-6,...
    %                 'Display', 'iter');
    %             problem = createOptimProblem('fmincon','x0',Init_scale,...
    %                     'objective',f_min,'lb',lb,'ub',ub,'nonlcon',nonlincon_fun);
    %            [xmin,fmin] = run(ms,problem, 40);
    
    % initialisation for first week
    best_fval = 0;
    f_vec = 0;
    if (fval<4*best_fval) && (fval<1.5*median(f_vec))
        good_i =i;
    else
        if useRealVola
            % if results are bad, use other h0
            vola_idx = vola_idx+1;
            while ((fval>=4*best_fval) || (fval>=1.5*median(f_vec))) && vola_idx<=num_voladays
                if vola_vec(vola_idx)~=0
                    if vola_idx==2
                        txt_msg =strcat('Bad optimization results. Trying yesterdays realized vola.');
                    else
                        txt_msg =strcat('Bad optimization results. Trying ',num2str(vola_idx-1),'-days prio realized vola.');
                    end
                    warning(txt_msg)
                    new_vola = vola_vec(vola_idx);
                    f_min2 = @(params) f_min_raw(params, scaler,new_vola);
                    nonlincon_fun2 = @(params) nonlincon_scale_v2(params, scaler);
                    [xxval2,fval2,exitflag2] =  fmincon(f_min2, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
                    if fval2<fval
                        xxval =xxval2;
                        fval = fval2;
                        exitflag = exitflag2;
                        sig2_0(i) = new_vola;
                    end
                end
                vola_idx =vola_idx+1;
            end
        else
            warning('Bad optimization results. No other starting values or optimization methods implemented for h0 calibration so far. Come back later ;)')
        end
        if ((fval>=4*best_fval) || (fval>=1.5*median(f_vec)))
            warning('Bad optimization results. No other starting values left! Continue with next week.')
            bad_i(end+1) =i;
        else
            good_i(end+1)=i;
        end
    end
    opt_params_raw(i, :) = xxval;
    struc.optispecs.flag = exitflag;
    struc.optispecs.goalval = fval;
    opt_params_clean(i, :) = opt_params_raw(i, :).*scaler;
    scale_tmp           =   magnitude(opt_params_clean(i, :));
    if ~(useRealVola || useMLEPh0)
        sig2_0(i) = opt_params_clean(i, 5);
    end
    if i == min(weeksprices)
        scaler_firstweek= scale_tmp;
        f_val_firstweek = fval;
        f_vec = fval;
        best_fval = fval;
        best_x = opt_params_clean(i, :);
        best_scaler = scale_tmp;
        param_vec = opt_params_clean(i, :);
        num_vec = length(data_week);
    else
        f_vec(end+1) =fval;
        param_vec(end+1,:) = opt_params_clean(i, :);
        num_vec(end+1) = length(data_week);
        if fval<best_fval
            best_fval = fval;
            best_scaler =scale_tmp;
            best_x = opt_params_clean(i, :);
        end
    end
    
    param_vec_weekly(i,:) =  opt_params_clean(i, :);
    struc.optispecs.scaler         =   scale_tmp;
    if useRealVola
        struc.vola_idx      =   vola_idx;
    end
    struc.sig20         =   sig2_0(i);
    struc.hngPrice      =   abs(price_Q(opt_params_clean(i,:), data_week, r_cur./252, sig2_0(i))) ;
    struc.blsimpvhng    =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, struc.hngPrice');
    struc.epsilonhng    =   (struc.Price - struc.hngPrice) ./  struc.blsvega';
    struc.epsilonbls    =   (struc.Price - struc.blsPrice) ./  struc.blsvega';
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
    
    disp(struc.MSE);
    disp(struc.hngparams(3));
end
if strcmp(algorithm,'interior-point') %for file naming purposes
    algorithm = 'interiorpoint';
end
if useRealVola
    save(strcat('params_options_',num2str(year),'_h0asRealVolaGS',num2str(num_voladays),'days_',goal,'_',algorithm,'_',txt,'.mat'),'values');
elseif useMLEPh0
    save(strcat('params_options_',num2str(year),'_h0ashtMLEPGS_',goal,'_',algorithm,'_',txt,'_m.mat'),'values');
else
    save(strcat('params_options_',num2str(year),'_h0_calibratedGS_',goal,'_',algorithm,'_',txt,'f.mat'),'values');
end
%for specific weeks
%save(strcat('params_Options_',num2str(year),'week2and4','_h0asRealVola_',goal,'_',algorithm,'_',txt,'.mat'),'values');
