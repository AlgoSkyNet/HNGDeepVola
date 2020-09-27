clc;
clearvars;
close all;
warning('on')
datatable       = readtable('SP500_220320.csv');
dataRet         = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
win_len         = 2520; % around 10years
years           = (dataRet(:,2)==2013);% | (data(:,2)==2011) | (data(:,2)==2012) | (data(:,2)==2013) | (data(:,2)==2014) | (data(:,2)==2015) | (data(:,2)==2016) | (data(:,2)==2017) | (data(:,2)==2018);
wednesdays      = (weekday(dataRet(:,1))==4);
doi             = years & wednesdays; %days of interest
index           = find(doi);
shortdata       = dataRet(doi,:);
tmp             = shortdata(1,2)-2012; %  year
display(datatable.Date(index(1)));





%parpool()
%path                = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
%path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
path                =  'C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';
year                = 2013;
useYield            = 0; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 1; % use last h_t from MLE under P as h0
num_voladays        = 6; % if real vola, give the number of historic volas used (6 corresponds to today plus 5 days = 1week);
algorithm           = 'interior-point';% 'sqp'
goal                =  'MSE'; % 'MSE';   'MAPE';  ,'OptLL';
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

% start from the first Wednesday of 2010 and finish with the last Wednesday
% of 2010
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
%load(strcat('C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year),'_mle_opt_h0est.mat'));
%load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year),'_mle_opt_h0est.mat'));
load(strcat('/Users/lyudmila/Documents/GitHub/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year),'_mle_opt_h0est.mat'));

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

if useMLEPh0
    f_min_raw = @(params,scaler,sig2_0) runCalibration(params.*scaler, weeksprices, data, sig_tmp(2), SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index);
else
    f_min_raw = @(params,scaler) runCalibrationh0(params.*scaler, weeksprices, data, SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index);
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
    'TolFun', 1e-9,...
    'TolX', 1e-9,...
    'TypicalX', Init(2,:)./scaler);%, 'UseParallel', 'always');


%local optimization
[xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
xmin_fmincon = xxval.*scaler;
params = xmin_fmincon;

[fValOut, values]=getCalibratedData(params, weeksprices, data, sig_tmp(2), SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index);
save('res2013_h0P_r.mat');

% %local optimization
% [xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
% xmin_fmincon = xxval.*scaler;
% 
% save('res2014_h0P.mat');
%  gs = GlobalSearch('XTolerance',1e-9,'FunctionTolerance', 1e-9,...
%             'StartPointsToRun','bounds-ineqs','NumTrialPoints',2e3,'Display','final');
%  problem = createOptimProblem('fmincon','x0',Init_scale,...
%                 'objective',f_min,'lb',lb,'ub',ub,'nonlcon',nonlincon_fun);
% [xmin,fmin] = run(gs,problem);
% 
% xmin_gs = xmin.*scaler;
