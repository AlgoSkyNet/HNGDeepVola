%HNG-Optimization under Q
%Options: path = '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
clc;
clearvars;
close all;


%parpool()
path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';
year                = 2010;
useYield            = 0; % uses tbils now
useRealVola         = 1; % alwas use realized vola
global_idx          = 0; % indicator if globalisation algorithm should be used.
% Globalisation is not recommend; little
% improvement for huge computitional effort
global_stage2       = 0; % indicator if globalisation algorihtm should be used as second order solution
algorithm           = "interior-point";% "sqp"
goal                =  "MSE"; % "MSE";   "MAPE";  ,"OptLL";
path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);

% load Interest rates
% load the corresponding data
if useYield
    path_vola           =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
    txt = "useYield";
else
    path_vola           =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
    txt = "noYield";
end
load(path_vola);

% if use realized volatility data then load the corresponding data
bound                   = [100, 100];
formatIn                = 'dd-mmm-yyyy';

% start from the first Wednesday of the current year and finish with the
% last Wednesday of the same year

DateString_start        = strcat('01-January-',num2str(year));
DateString_end          = strcat('31-December-',num2str(year));
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
wednessdays             = (weekday(date_start:date_end)==4);
Dates                   = date_start:date_end;
Dates                   = Dates(wednessdays);

% initialize with the data from MLE estimation for each week
path                    =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/';
load(strcat(path,'weekly_',num2str(year),'_mle_opt.mat'));
Init                    = params_tmp;
% bounds for maturity, moneyness, volumes, interest rates
Type                    = 'call';
MinimumVolume           = 100;
MinimumOpenInterest     = 100;
IfCleanNans             = 1;
TimeToMaturityInterval  = [8, 250];
MoneynessInterval       = [0.9, 1.1];
% load options data from the current year using the bounds above
[OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptions(Dates, Type, ...
    TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans,...
    TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume, ...
    OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
    TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);
% get the numbers of weeks for which the options data is available
weeksprices             = week(datetime([OptionsStruct.date], 'ConvertFrom', 'datenum'));

% get a list of unique week numbers
idxj  = 1:length(unique(weeksprices));

% construct a dataset from the OptionsStruct
data = [OptionsStruct.price; ...
    OptionsStruct.maturity; ...
    OptionsStruct.strike; ...
    OptionsStruct.priceunderlying; ...
    OptionsStruct.vega; ...
    OptionsStruct.implied_volatility];

%% Optimization

% Initialization
% initial scale is determined by the magnitude of initial point coming from
% the MLE under P estimation
sc_fac           =   magnitude(Init);
Init_scale_mat   =   Init./sc_fac;
% bounds for parameters [omega, akpha, beta, gamma*]
lb_mat           =   [1e-12, 0, 0, -1500];
ub_mat           =   [1, 1, 1, 1500];
% number of calibration parameters
num_clbr_params  = 4;
opt_params_raw   =   zeros(max(weeksprices), num_clbr_params);
opt_params_clean =   zeros(max(weeksprices), num_clbr_params);
values           =   cell(1, max(weeksprices));
sig2_0           =   zeros(1, max(weeksprices));

% values in first iteration:
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
    
    % initialize structure where all the parameters are going to be stored
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
    % handle negative interest rates
    % the convention is that we set them to zero since the observed values
    % are very close to zero
    for k = 1:length(interestRates)
        if interestRates(k) < 0
            interestRates(k) = 0;
        end
    end
    
    % compute the interest rates over the life of the options using the
    % data coming from the yield or Tbill data
    r_cur = zeros(length(data_week), 1);
    if useYield
        for k = 1:length(data_week)
            if data_week(k, 2) < 21
                r_cur(k) = interestRates(1);
            else
                r_cur(k) = interp1([21,63,126,252]./252, interestRates, data_week(k, 2)./252);
            end
        end
    else
        for k = 1:length(data_week)
            if data_week(k, 2) < 21 && ~isnan(interestRates(1))
                r_cur(k) = interestRates(1);
            else
                notNaN = ~isnan(interestRates);
                daylengths = [21, 42, 13*5, 126, 252]./252;
                r_cur(k) = interp1(daylengths(notNaN), interestRates(notNaN), data_week(k, 2)./252);
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
    if strcmp(goal, "MSE")
        f_min_raw = @(params, scaler) (mean((price_Q(params.*scaler, data_week, r_cur./252, sig2_0(i))' - data_week(:, 1)).^2));
        % MRAE/MAPE
    elseif strcmp(goal, "MAPE")
        f_min_raw = @(params,scaler) mean(abs(price_Q(params.*scaler, data_week, r_cur./252, sig2_0(i))' - data_week(:, 1))./data_week(:, 1));
        % Option Likelyhood
    elseif strcmp(goal ,"OptLL")
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
    % Starting value check
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
        [init_f, min_i]    = min([f1, f2, f3]);
        if min_i == 1
            Init_scale  = x1;
            scaler      = sc_fac(i, :);
            disp(strcat("Initial value used 'MLE parameters'."));
        elseif min_i == 2
            Init_scale  = x2;
            scaler      = scale_tmp;
            disp(strcat("Initial value used 'previous week'."));
        elseif min_i ==3
            Init_scale  = x3;
            scaler      = scaler_firstweek;
            disp(strcat("Initial value used 'first week'."));
        end
    else
        disp(strcat("Initial value used 'MLE parameters'."));
    end
    % fun2opti,scaled
    f_min           = @(params) f_min_raw(params, scaler);
    % constraint,scaled
    nonlincon_fun   = @(params) nonlincon_scale_v2(params, scaler);
    %parameter bounds, scaled
    lb              = lb_mat./scaler;
    ub              = ub_mat./scaler;
    %optimization specs
    opt = optimoptions('fmincon', ...
        'Display', 'iter', ...
        'Algorithm', algorithm, ...
        'MaxIterations', 3000,...
        'MaxFunctionEvaluations', 1500, ...
        'TolFun', 1e-6, ...
        'TolX', 1e-6, ...
        'TypicalX', Init(i, :)./scaler);
    
    struc.optispecs = struct();
    struc.optispecs.optiopt = opt;
    struc.optispecs.scaleproblem = 0;
    
    %use function scaling if function values are big
    if magnitude(init_f) > 100
        opt.ScaleProblem = 'obj-and-constr' ;
        struc.optispecs.scaleproblem = 1;
    end
    
    if global_idx
        %Global optimization
        gs = GlobalSearch('XTolerance', 1e-6, ...
            'FunctionTolerance', 1e-2, ...
            'StartPointsToRun', 'bounds-ineqs', ...
            'Display','iter', ...
            'NumTrialPoints', 1000, ...
            'NumStageOnePoints', 950);
        problem = createOptimProblem('fmincon', 'x0', Init_scale, ...
            'objective', f_min, 'lb', lb, 'ub', ub, 'nonlcon', nonlincon_fun, 'options', opt);
        [xxval, fval, exitflag, gsresults, gsvec] = run(gs, problem);
        struc.optispecs.gsopt = gs;
        struc.optispecs.gsresults = gsresults;
        struc.optispecs.localminima = gsvec;
    else
        %local optimization
        [xxval, fval, exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    end
    
    if (i== min(weeksprices)) || ((fval < 2 * f_val_firstweek) && fval < 1.5 * median(f_vec)) || (min_i == 1)
    else
        % if optimization results are bad and MLE were not used as initial
        % parameters check those values
        warning("Bad optimization results. Trying MLE Parameters...")
        Init_scale2     = Init_scale_mat(i, :);
        scaler2         = sc_fac(i, :);
        f_min2          = @(params) f_min_raw(params, scaler);
        nonlincon_fun2  = @(params) nonlincon_scale_v2(params, scaler);
        lb2             = lb_mat./scaler;
        ub2             = ub_mat./scaler;
        opt.TypicalX    = Init(i,:)./scaler;
        
        [xxval2, fval2, exitflag2] =  fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
        if fval2 < fval
            xxval = xxval2;
            fval = fval2;
            exitflag = exitflag2;
            scaler = scaler2;
        end
    end
    % if optimization results are still bad, use globalisation algorithm (if not already used)
    if(i ~= min(weeksprices)) && global_stage2
        if (global_idx == 0) && ((fval > 2 * f_val_firstweek) || fval > 1.5 * median(f_vec))
            warning("Bad optimization results. Globalisation initialised...")
            gs = GlobalSearch('XTolerance', 1e-6, ...
                    'FunctionTolerance', 1e-2, ...
                    'StartPointsToRun', 'bounds-ineqs', ...
                    'Display', 'iter', ...
                    'NumTrialPoints', 1000,...
                    'NumStageOnePoints', 950);
            x_gloopt        = xxval.*scaler;
            scaler          = magnitude(x_gloopt);
            
            f_min           = @(params) f_min_raw(params, scaler);
            nonlincon_fun   = @(params) nonlincon_scale_v2(params, scaler);
            Init_scale      = x_gloopt./scaler;
            lb              = lb_mat./scaler;
            ub              = ub_mat./scaler;
            
            % trying to get huge improvements
            opt2 = optimoptions('fmincon', ...
                'Display', 'iter', ...
                'Algorithm', algorithm, ...
                'MaxIterations', 300, ...
                'MaxFunctionEvaluations', 1500, ...
                'TolFun', 1e-2, ...
                'TolX', 1e-6, ...
                'TypicalX', Init(i, :)./scaler);
            
            opt2.TypicalX = Init(i, :)./scaler;
            problem = createOptimProblem('fmincon', 'x0', Init_scale, ...
                'objective', f_min, 'lb', lb, 'ub', ub, 'nonlcon', nonlincon_fun, 'options', opt2);
            [xxval, fval, exitflag, gsresults, gsvec] = run(gs, problem);
            struc.optispecs.gsopt   = gs;
            struc.optispecs.gsresults = gsresults;
            struc.optispecs.localminima = gsvec;
        end
        %         ms = MultiStart('FunctionTolerance', 1e-2, 'StartPointsToRun', 'bounds-ineqs');
        %          [xxval3, fval3, exitflag3] =run(ms, problem, 50000);
        %         Init_scale = mean(Init);
        %         scaler = magnitude(Init_scale);
        %         f_min = @(params) f_min_raw(params,  scaler);
        %         nonlincon_fun = @(params) nonlincon_scale_v2(params,  scaler);
        %         lb = lb_mat./scaler;
        %         ub = ub_mat./scaler;
        %         problem = createOptimProblem('fmincon', 'x0', Init_scale, ...
        %             'objective', f_min, 'lb', lb, 'ub', ub, 'nonlcon', nonlincon_fun, 'options', opt);
        %         tpoints = CustomStartPointSet(mvnrnd(Init_scale, cov(Init./scaler), 1000));
        %         tpts = list(tpoints);
        %         isequal(ptmatrix, tpts)
        %         [xxval3, fval3, exitflag3]=run(ms, problem, tpoints)
    end
    
    opt_params_raw(i, :) = xxval;
    struc.optispecs.flag = exitflag;
    struc.optispecs.goalval = fval;
    opt_params_clean(i, :) = opt_params_raw(i, :).*scaler;
    scale_tmp           =   magnitude(opt_params_clean(i, :));
    if i == min(weeksprices)
        scaler_firstweek= scale_tmp;
        f_val_firstweek = fval;
        f_vec = fval;
        param_vec = opt_params_clean(i, :);
        num_vec = length(data_week);
    else
        f_vec(end+1) =fval;
        param_vec(end+1,:) = opt_params_clean(i, :);
        num_vec(end+1) = length(data_week);
    end
    
    struc.optispecs.scale         =   scale_tmp;
    struc.hngPrice      =   price_Q(opt_params_clean(i, :), data_week, r_cur./252, sig2_0(i)) ;
    struc.blsimpvhng    =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, struc.hngPrice');
    struc.epsilonhng    =   (struc.Price - struc.hngPrice) ./ data_week(:, 5)';
    struc.epsilonbls    =   (struc.Price - struc.blsPrice) ./ data_week(:, 5)';
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
if strcmp(algorithm, "interior-point") %for file naming purposes
    algorithm = "interiorpoint";
end
save(strcat('params_Options_', num2str(year), '_h0asRealVola_', goal, '_', algorithm, '_', txt, '.mat'), 'values');

%for specific weeks
%save(strcat('params_Options_', num2str(year), 'week2and4', '_h0asRealVola_', goal, '_', algorithm, '_', txt, '.mat'), 'values');
%%
analysis = 0;
j = j + 1;
if analysis
    figure("Name", num2str(year))
    ind_temp = 0;
    bad_idx = (f_vec > quantile(f_vec, 0.8));
    bad_weeks = unique(weeksprices);
    bad_weeks = bad_weeks(bad_idx);
    bad_weeks = [2, 5];
    num_vec = [10, 10];
    weeksplot = bad_weeks; %unique(weekprices);
    for i = weeksplot
        ind_temp = ind_temp + 1;
        data_week = data(:, (weeksprices == i))';
        k = ceil(sqrt(length(weeksplot)));
        subplot(k, k, j)
        [xq, yq] = meshgrid(unique(data_week(:, 2)), unique(data_week(:, 3)));
        vq = griddata(data_week(:, 2), data_week(:, 3), data_week(:, 1), xq, yq);  
        %(x, y, v) being your original data for plotting points
        surf(xq, yq, vq)
        hold on
        scatter3(data_week(:, 2), data_week(:, 3), data_week(:, 1), "o", "k", 'filled');
        ylim([1000 1300])
        xlim([0 250])
        view(0, 90)
        colormap(jet(256));
        caxis([0, 100]);
        colorbar
        title(strcat("week ", num2str(i)))
    end
    corr_  = corr(param_vec);
    mean_ = mean(param_vec);
    median_ = median(param_vec);
    corr_size_err = corr(f_vec, num_vec);
    
    
end