clc;
clearvars;
close all;
warning('on')
for iii=[2010:2018]
ifHalfYear      = 1;
currentYear     = iii;
datatable       = readtable('SP500_220320.csv');
dataRet         = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
win_len         = 2520; % around 10years
years           = (dataRet(:,2) == currentYear);
wednesdays      = (weekday(dataRet(:,1)) == 4);
if ifHalfYear
    months      = (month(dataRet(:,1)) == 7 | month(dataRet(:,1))==8 | month(dataRet(:,1))==9 | month(dataRet(:,1))==10 | month(dataRet(:,1))==11 | month(dataRet(:,1))==12);
    doi         = years & months & wednesdays; %days of interest
else
    doi         = years & wednesdays; %days of interest
end
index           = find(doi);
shortdata       = dataRet(doi,:);
display(datatable.Date(index(1)));


path                =  'C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';
year                = currentYear;
useYield            = 0; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 0; % use last h_t from MLE under P as h0
useUpdatedh0Q       = 0; % use last h_t from MLE under P for 10 years, then updated under Q for one more year
path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);
if ifHalfYear
    if useMLEPh0
    load(strcat('res', num2str(currentYear), '_h0P_6m.mat'),'values');
elseif useUpdatedh0Q
    load(strcat('res', num2str(currentYear), '_h0Q_6m.mat'),'values');
elseif useRealVola
    load(strcat('res', num2str(currentYear), '_h0RV_6m.mat'),'values');
else
    load(strcat('res', num2str(currentYear), '_6m.mat'),'values');
    
end

else
if useMLEPh0
    load(strcat('res', num2str(currentYear - 1), '_h0P.mat'),'values');
elseif useUpdatedh0Q
    load(strcat('res', num2str(currentYear - 1), '_h0Q.mat'),'values');
elseif useRealVola
    load(strcat('res', num2str(currentYear - 1), '_h0RV.mat'),'values');
else
    load(strcat('res', num2str(currentYear - 1), '.mat'),'values');
    
end
end
%
%

hngparams = values{1,length(values)}.hngparams(1:4);
sig20 = values{1,length(values)}.sig20;


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
    DateString_start      = strcat('01-July-',num2str(year));
end
DateString_end      = strcat('31-December-',num2str(year));
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
wednessdays             = (weekday(date_start:date_end)==4);
Dates                   = date_start:date_end;
Dates                   = Dates(wednessdays);


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
idxj  = 1:length(unique(weeksprices));
indSigma = uniqueWeeks(1);


data = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega; OptionsStruct.implied_volatility];
% save('generaldata2015.mat', 'data', 'DatesClean', 'OptionsStruct', 'OptFeatures', 'idx');

load(strcat('/Users/lyudmila/Documents/GitHub/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year),'_mle_opt_h0est.mat'));


MSE = 0;
[fValOut, values]=getCalibratedData(hngparams, weeksprices, data, sig20, SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index);
if useMLEPh0
    for i = 1:length(values)
        if ~isempty(values{1,i})
            optLL_val_P(i)= values{1,i}.optionsLikhng;
            MSE_P(i)= values{1,i}.MSE;
            IVRMSE_P(i)= values{1,i}.IVRMSE;
        end
    end
    save(strcat('s', num2str(currentYear), '.mat'),'MSE_P', 'optLL_val_P','IVRMSE_P','-append' );
elseif useUpdatedh0Q
    for i = 1:length(values)
        if ~isempty(values{1,i})
            optLL_val_Q(i)= values{1,i}.optionsLikhng;
            MSE_Q(i)= values{1,i}.MSE;
            IVRMSE_Q(i)= values{1,i}.IVRMSE;
        end
    end
    save(strcat('s', num2str(currentYear), '.mat'),'MSE_Q', 'optLL_val_Q','IVRMSE_Q','-append' );
elseif useRealVola
    for i = 1:length(values)
        if ~isempty(values{1,i})
            optLL_val_RV(i)= values{1,i}.optionsLikhng;
            MSE_RV(i)= values{1,i}.MSE;
            IVRMSE_RV(i)= values{1,i}.IVRMSE;
        end
    end
    save(strcat('s', num2str(currentYear), '.mat'),'MSE_RV', 'optLL_val_RV','IVRMSE_RV','-append' );
else
    for i = 1:length(values)
        if ~isempty(values{1,i})
            optLL_val_est(i)= values{1,i}.optionsLikhng;
            MSE_est(i)= values{1,i}.MSE;
            IVRMSE_est(i)= values{1,i}.IVRMSE;
        end
    end
    save(strcat('s', num2str(currentYear), '.mat'),'MSE_est', 'optLL_val_est','IVRMSE_est','-append' );
    
end
clearvars -except 'iii';
end

