%HNG-Optimization under Q 
%Options: path = '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
clc; 
clearvars; 
close all;
warning('on')

%parpool()
path                = 'D:/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
%path                = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
%path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
%path                =  'C:/Users/TEMP/Documents/GIT/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';
year                = 2018;
useYield            = 0; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 0; % use last h_t from MLE under P as h0
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
load(strcat('D:/GitHub/MasterThesisHNGDeepVola/Code/Calibration MLE/','weekly_',num2str(year),'_mle_opt_h0est.mat'));

%load(strcat('C:/Users/TEMP/Documents/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/MLE_P estimation results/','weekly_',num2str(year),'_mle_opt.mat'));

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

       
%% weekly optimization
j = 1;
good_i =[];
bad_i =[];
MSE_vec =[];
MAPE_vec = [];
OPTLL_vec =[];
for i = unique(weeksprices)
    disp(strcat('Optimization (',goal ,') of week ',num2str(i),' in ',num2str(year),'. h_0 will be calibrated.'))     
    data_week = data(:,(weeksprices == i))';
    if isempty(data_week)
        disp(strcat('no data for week ',num2str(i),' in ',num2str(year),'!'))
        continue
    end

    
    struc               =  struct();
    struc.numOptions    =  length(data_week(:, 1));
    % compute interest rates for the weekly options
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

    for k = 1:length(interestRates)
        if interestRates(k)<0
            interestRates(k)=0;
        end
    end
    j = j + 1;
    r_cur = zeros(length(data_week), 1);
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
    sig2_0(i) = sig_tmp(i);
    struc.Price         =   data_week(:, 1)';
    struc.yields        =   interestRates;
    struc.blsPrice      =   blsprice(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, vola_tmp(i), 0)';
    struc.blsimpv       =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, data_week(:, 1));
    indNaN = find(isnan(struc.blsimpv));
    struc.num_NaN_implVols = length(indNaN);    
    struc.blsimpv(indNaN) = data_week(indNaN, 6);
    struc.blsvega = blsvega(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, struc.blsimpv(:));
    struc.sig20         =   sig2_0(i);
    struc.hngPrice      =   price_Q(params_tmp(i,:), data_week, r_cur./252, sig2_0(i)) ;
    struc.blsimpvhng    =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, struc.hngPrice');
    struc.epsilonhng    =   (struc.Price - struc.hngPrice) ./ data_week(:,5)';
    struc.epsilonbls    =   (struc.Price - struc.blsPrice) ./ data_week(:,5)';
    s_epsilon2hng       =   mean(struc.epsilonhng(:).^2);
    s_epsilon2bls       =   mean(struc.epsilonbls(:).^2);
    struc.optionsLikhng =   -.5 * struc.numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1);
    struc.optionsLikbls =   -.5 * struc.numOptions * (log(2 * pi) + log(s_epsilon2bls) + 1);
    struc.meanPrice     =   mean(data_week(:, 1));
    struc.hngparams     =   params_tmp(i, :);
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
    %disp(struc.MSE);
    %disp(struc.MAPE);
    MSE_vec(end+1) = struc.MSE;
    MAPE_vec(end+1) = struc.MAPE;
    OPTLL_vec(end+1) = struc.optionsLikhng;
end 
save(strcat('mle_calls_',num2str(year)),'MSE_vec','MAPE_vec','OPTLL_vec')
