clc;
clearvars;
close all;
warning('on')
ifHalfYear      = 0;
currentYear     = 2010;
datatable       = readtable('SP500_220320.csv');
dataRet         = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
win_len         = 2520; % around 10years
years           = (dataRet(:,2) == currentYear);
wednesdays      = (weekday(dataRet(:,1))==4);
tuesdays        = (weekday(dataRet(:,1))==3);
if ifHalfYear
    months      = (month(dataRet(:,1))==1 | month(dataRet(:,1))==2 | month(dataRet(:,1))==3 | month(dataRet(:,1))==4 | month(dataRet(:,1))==5 | month(dataRet(:,1))==6);
    doi         = years & months & wednesdays; %days of interest
else
    doi         = years & wednesdays; %days of interest
end
index           = find(doi);
display(datatable.Date(index(1)));

%next period
tuesdays        = (weekday(dataRet(:,1))==3);
if ifHalfYear
    monthFirst  = (month(dataRet(:,1)) == 7) ;
    doiNextPeriod   = years & monthFirst & tuesdays; %days of interest of the next period
else
    monthFirst  = (month(dataRet(:,1)) == 1) ;
    yearNext        = (dataRet(:,2) == currentYear + 1);
    doiNextPeriod   = yearNext & monthFirst & tuesdays; %days of interest of the next period
end


indexNextPeriod      = find(doiNextPeriod);
indexNextPeriodFirst = indexNextPeriod(1);
display(datatable.Date(index(1)));
display(datatable.Date(index(end)));
display(datatable.Date(indexNextPeriodFirst));

%path                =  'C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/Data/Datasets';
path                =  'C:/GIT/HenrikAlexJP/Data/Datasets';
%pathF                =  'C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/';
pathF                =  'C:/GIT/HenrikAlexJP/';
stock_ind           = 'SP500';
year                = currentYear;
useYield            = 1; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 0; % use last h_t from MLE under P as h0
useUpdatedh0Q       = 0; % use last h_t from MLE under P for 10 years, then updated under Q for one more year
useRPrescribed      = 1;
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

bound                   = [100, 100];
formatIn                = 'dd-mmm-yyyy';

% start from the first Wednesday and finish with the last Wednesday
DateString_start        = strcat('01-January-',num2str(year));
if ifHalfYear
    DateString_end      = strcat('30-June-',num2str(year));
else
    DateString_end      = strcat('31-December-',num2str(year));
end
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
wednessdays             = (weekday(date_start:date_end)==4);
DatesYear               = date_start:date_end;
Dates                   = DatesYear(wednessdays);




% initialize with the data from MLE estimation for each week
if useUpdatedh0Q
    load(strcat(pathF,'Code/calibration checks/Calibration MLE P/Results with estimated h0P for Update/','weekly_',num2str(year),'_mle_opt_h0est_UpdateQ.mat'));
elseif useRPrescribed
    load(strcat(pathF,'Code/calibration checks/Calibration MLE P/Results with estimated h0p rAv/','weekly_',num2str(year),'_mle_opt_h0est_rAv.mat'));
else
    load(strcat(pathF,'Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year),'_mle_opt_h0est.mat'));
    
end
if useRealVola || useMLEPh0 || useUpdatedh0Q
    num_params = 4;
else
    num_params = 5;
end

Init = params_tmp;
if ~(useRealVola || useMLEPh0 || useUpdatedh0Q)
    Init = [params_tmp,sig_tmp];
end

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
uniqueWeeks = unique(weeksprices);
indSigma = uniqueWeeks(1);


data = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega; OptionsStruct.implied_volatility];
%% Optimization
% Initialization
sc_fac           =   magnitude(Init);
Init_scale_mat   =   Init./sc_fac;
lb_mat           =   [1e-12, 0, 0, -1300];
ub_mat           =   [1, 1, 1, 1300];
algorithm           = 'interior-point';% 'sqp'

if ~(useRealVola || useMLEPh0 || useUpdatedh0Q)
    lb_mat = [lb_mat, 1e-12];
    ub_mat = [ub_mat, 1];
end
opt_params_raw   =   zeros(max(weeksprices), num_params);
opt_params_clean =   zeros(max(weeksprices), num_params);
sig2_0           =   zeros(1,max(weeksprices));

%values in first iteration:
Init_scale       =   Init_scale_mat(min(weeksprices), :);
scaler           =   sc_fac(min(weeksprices), :);
if useYield
    indYearly = 8;
else
    indYearly = 9;
end
if useRPrescribed
    r_all_yearly = SP500_date_prices_returns_realizedvariance_interestRates(indYearly, ...
        SP500_date_prices_returns_realizedvariance_interestRates(1,:) >= DatesYear(1) &...
        SP500_date_prices_returns_realizedvariance_interestRates(1,:) <= DatesYear(end));
    rValue = nanmean(r_all_yearly);
else
    rValue = 0;
end

if useMLEPh0 || useUpdatedh0Q
    f_min_raw = @(params,scaler,sig2_0) runCalibration(params.*scaler, weeksprices, data, sig_tmp(indSigma), SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index);
elseif useRealVola
    sig_tmp = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
        SP500_date_prices_returns_realizedvariance_interestRates(1,:)== dataRet(index(1),1));
    f_min_raw = @(params,scaler,sig2_0) runCalibration(params.*scaler, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index);
    
else
    f_min_raw = @(params,scaler) runCalibrationh0(params.*scaler, weeksprices, data, SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index, rValue);
end
f_min = @(params) f_min_raw(params(1:num_params), scaler);
% constraint,scaled
nonlincon_fun = @(params) nonlincon_scale_v2(params, scaler);
%parameter bounds, scaled
lb = lb_mat./scaler;
ub = ub_mat./scaler;
%optimization specs
opt = optimoptions('fmincon', ...
    'Display', 'iter',...
    'Algorithm', algorithm,...
    'MaxIterations', 3000,...
    'MaxFunctionEvaluations',20000, ...
    'TolFun', 1e-15,...
    'TolX', 1e-15,...
    'TypicalX', Init(2,:)./scaler);%, 'UseParallel', 'always');

if useMLEPh0 || useUpdatedh0Q
    %local optimization
    [xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    xmin_fmincon = xxval.*scaler;
    params = xmin_fmincon;
    [fValOut, values] = getCalibratedData(params, weeksprices, data, sig_tmp(indSigma), SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index);
    
    logret = dataRet(index(1):indexNextPeriodFirst,4);
    [~, sigmaseries] = ll_hng_Q_n(params(1:4), logret, rValue, sig_tmp(indSigma));
    sigma20forNextPeriod = sigmaseries(last);
elseif useRealVola
    %local optimization
    %     [xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    %     xmin_fmincon = xxval.*scaler;
    %     params = xmin_fmincon;
    %     [fValOut, values]=getCalibratedData(params, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index);
    gs = GlobalSearch('XTolerance',1e-9,'FunctionTolerance', 1e-9,...
        'StartPointsToRun','bounds-ineqs','NumTrialPoints',1e3,'Display','final');
    problem = createOptimProblem('fmincon','x0',Init_scale,...
        'objective',f_min,'lb',lb,'ub',ub,'nonlcon',nonlincon_fun);
    [xmin,fmin] = run(gs,problem);
    xmin_gs = xmin.*scaler;
    params = xmin_gs;
    [fValOut, values] = getCalibratedData(params, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index);
    
    logret = dataRet(index(1):indexNextPeriodFirst,4);
    [~, sigmaseries] = ll_hng_Q_n(params(1:4), logret,rValue, sig_tmp);
    sigma20forNextPeriod = sigmaseries(last);
else
    %local optimization
    [xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    xmin_fmincon = xxval.*scaler;
    params = xmin_fmincon;
    [fValOut, values] = getCalibratedDatah0(params, weeksprices, data, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index, rValue);
    
    logret = dataRet(index(1):indexNextPeriodFirst,4);
    [~, sigmaseries] = ll_hng_Q_n(params(1:4),logret,rValue,params(5));
    sigma20forNextPeriod = sigmaseries(last);
end

strYear = num2str(currentYear);
if useYield
    flagYield = '_yield';
else
     flagYield = '_tbill';
end

if useRPrescribed
    flagR = '_avR';
else
    flagR = '';
end
if ifHalfYear
    flagNmonths = '_6m';
else
    flagNmonths = '_12m';
end
if useMLEPh0
    save(strcat('res', strYear, '_h0P', flagNmonths, flagR, flagYield, '.mat'));
elseif useUpdatedh0Q
    save(strcat('res', strYear, '_h0Q', flagNmonths, flagR, flagYield, '.mat'));
elseif useRealVola
    save(strcat('res', strYear, '_h0RV', flagNmonths, flagR, flagYield, '.mat'));
else
    save(strcat('res', strYear, '_h0calibr', flagNmonths, flagR, flagYield, '.mat'));
end
