function [] = calibrateMO(useServer, ifHalfYear, currentYear, useYield, useScenario, useAverageWhenCal, fileNameWithMLEPests)

datatable       = readtable('SP500_220320.csv');
dataRet         = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
years           = (dataRet(:,2) == currentYear);
wednesdays      = (weekday(dataRet(:,1))==4);
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
    fileNameFlagNmonths = '_6m';
else
    monthFirst  = (month(dataRet(:,1)) == 1) ;
    yearNext        = (dataRet(:,2) == currentYear + 1);
    doiNextPeriod   = yearNext & monthFirst & tuesdays; %days of interest of the next period
    fileNameFlagNmonths = '_12m';
end


indexNextPeriod      = find(doiNextPeriod);
indexNextPeriodFirst = indexNextPeriod(1);
display(datatable.Date(index(1)));
display(datatable.Date(index(end)));
display(datatable.Date(indexNextPeriodFirst));

switch useServer
    case 1
        pathPrefix = 'C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/';
    case 2
        pathPrefix = 'C:/GIT/HenrikAlexJP/';
    otherwise
        pathPrefix = '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/';
end
if useYield
    path_r       =  strcat(pathPrefix, 'Data/Datasets/InterestRates/SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
    fileNameTyper = '_yield';
    fileNameFlagR = '_rAverage';
    ind_r = 8;
   
else
    path_r       =  strcat(pathPrefix, 'Data/Datasets/InterestRates/SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
    fileNameTyper = '_tbill';
    fileNameFlagR = '_rFirstWeek';
    ind_r = 9;
end

stock_ind           = 'SP500';
path_               = strcat(pathPrefix, 'Data/Datasets/', stock_ind, '/', 'Calls', num2str(currentYear), '.mat');
% load Interest rates
% load the corresponding data
load(path_);
load(path_r);

formatIn                = 'dd-mmm-yyyy';
% start from the first Wednesday and finish with the last Wednesday of
% the period
DateString_start        = strcat('01-January-',num2str(currentYear));
if ifHalfYear
    DateString_end      = strcat('30-June-',num2str(currentYear));
else
    DateString_end      = strcat('31-December-',num2str(currentYear));
end
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
wednessdays             = (weekday(date_start:date_end)==4);
DatesYear               = date_start:date_end;
Dates                   = DatesYear(wednessdays);


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
% Optimization
% Initialization
lb_mat           =   [1e-12, 0, 0, -1300];
ub_mat           =   [1, 1, 1, 1300];
algorithm           = 'interior-point';



num_params = 4;
fileNameEnd = [];
switch useScenario
    case 1
        useMLEPh0           = 1; % use last h_t from MLE under P as h0
        fileNameScenarioType = '_h0P';
    case 2
        useRealVola         = 1;
        fileNameScenarioType = '_h0RV';
    case 3
        useUpdatedh0Q       = 1;
        fileNameScenarioType = '_h0Q';
    otherwise
        num_params = 5;
        
        fileNameScenarioType = '_h0calibr';
end


load(fileNameWithMLEPests);


Init = params_tmp;
if useScenario == 4
    Init = [params_tmp,sig_tmp];
        lb_mat = [lb_mat, 1e-12];
        ub_mat = [ub_mat, 1];
end


sc_fac           =   magnitude(Init);
Init_scale_mat   =   Init./sc_fac;
%values in first iteration:
Init_scale       =   Init_scale_mat(min(weeksprices), :);
scaler           =   sc_fac(min(weeksprices), :);

if useAverageWhenCal
    r_all_yearly = SP500_date_prices_returns_realizedvariance_interestRates(ind_r, ...
        SP500_date_prices_returns_realizedvariance_interestRates(1,:) >= DatesYear(1) &...
        SP500_date_prices_returns_realizedvariance_interestRates(1,:) <= DatesYear(end));
    rValue = nanmean(r_all_yearly);
else
    rValue = SP500_date_prices_returns_realizedvariance_interestRates(ind_r, SP500_date_prices_returns_realizedvariance_interestRates(1,:) == DatesYear(1));
end

if useMLEPh0 || useUpdatedh0Q
    f_min_raw = @(params,scaler,sig2_0) runCalibration(params.*scaler, weeksprices, data, sig_tmp(indSigma), SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index, rValue);
elseif useRealVola
    sig_tmp = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
        SP500_date_prices_returns_realizedvariance_interestRates(1,:)== dataRet(index(1),1));
    curIndex = dataRet(index(1),1) - 1;
    while isempty(sig_tmp)
        sig_tmp = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:)== curIndex);
        curIndex = curIndex - 1;
    end
    f_min_raw = @(params,scaler,sig2_0) runCalibration(params.*scaler, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index, rValue);
    
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
%local optimization

[xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
xmin_fmincon = xxval.*scaler;
params = xmin_fmincon;
%     gs = GlobalSearch('XTolerance',1e-9,'FunctionTolerance', 1e-9,...
%         'StartPointsToRun','bounds-ineqs','NumTrialPoints',1e3,'Display','final');
%     problem = createOptimProblem('fmincon','x0',Init_scale,...
%         'objective',f_min,'lb',lb,'ub',ub,'nonlcon',nonlincon_fun);
%     [xmin,fmin] = run(gs,problem);
%     xmin_gs = xmin.*scaler;
%     params = xmin_gs;
%     [fValOut, values] = getCalibratedData(params, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index, rValue);
logret = dataRet(index(1):indexNextPeriodFirst,4);

if useMLEPh0 || useUpdatedh0Q
    [fValOut, values] = getCalibratedData(params, weeksprices, data, sig_tmp(indSigma), SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index, rValue);
    sigmaseries = sim_hng_Q_n(params(1:4), logret, rValue, sig_tmp(indSigma));
elseif useRealVola
    [fValOut, values] = getCalibratedData(params, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index, rValue);
    sigmaseries = sim_hng_Q_n(params(1:4), logret, rValue, sig_tmp);
else
    [fValOut, values] = getCalibratedDatah0(params, weeksprices, data, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index, rValue);
    sigmaseries = sim_hng_Q_n(params(1:4),logret,rValue,params(5));
end
sigma20forNextPeriod = sigmaseries(end);

save(strcat('res', num2str(currentYear), fileNameScenarioType, fileNameFlagNmonths, fileNameFlagR, fileNameTyper, '.mat'));
end

