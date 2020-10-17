clc;
clearvars;
close all;
warning('on')
for iii=[2011]
ifHalfYear      = 0;
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
useYield            = 1; % uses tbils now
useRealVola         = 1; % alwas use realized vola
useMLEPh0           = 0; % use last h_t from MLE under P as h0
useUpdatedh0Q       = 0; % use last h_t from MLE under P for 10 years, then updated under Q for one more year
path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);
if ifHalfYear
    if useMLEPh0
    load(strcat('res', num2str(currentYear), '_h0P_6m_avR_yield.mat'),'values', 'sigma20forNextPeriod','rValue');
elseif useUpdatedh0Q
    load(strcat('res', num2str(currentYear), '_h0Q_6m_avR_yield.mat'),'values', 'sigma20forNextPeriod','rValue');
elseif useRealVola
    load(strcat('res', num2str(currentYear), '_h0RV_6m_avR_yield.mat'),'values', 'sigma20forNextPeriod','rValue');
else
    load(strcat('res', num2str(currentYear), '_h0calibr_6m_avR_yield.mat'),'values', 'sigma20forNextPeriod','rValue');
    
end

else
if useMLEPh0
    load(strcat('res', num2str(currentYear - 1), '_h0P_12m_avR_yieldf.mat'),'values', 'sigma20forNextPeriod','rValue');
elseif useUpdatedh0Q
    load(strcat('res', num2str(currentYear - 1), '_h0Q_12m_avR_yieldf.mat'),'values', 'sigma20forNextPeriod','rValue');
elseif useRealVola
    load(strcat('res', num2str(currentYear - 1), '_h0RV_12m_avR_yieldf.mat'),'values', 'sigma20forNextPeriod','rValue');
else
    load(strcat('res', num2str(currentYear - 1), '_h0calibr_12m_avR_yieldf.mat'),'values', 'sigma20forNextPeriod','rValue');
    
end
end
%
%

hngparams = values{1,length(values)}.hngparams(1:4);
% sig20 = values{1,length(values)}.sig20;
sig20 = sigma20forNextPeriod;

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

[OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptionsFilt(Dates, Type, ...
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

load(strcat('/Users/lyudmila/Documents/GitHub/HenrikAlexJP/Code/calibration checks/Calibration MLE P/correct Likelihood/Yields/Results with estimated h0P rAv/','weekly_',num2str(year),'_mle_opt_h0est_rAv.mat'));


MSE = 0;
[fValOut, values]=getCalibratedData(hngparams, weeksprices, data, sig20, SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index, rValue);
if useMLEPh0
    j = 1;
    for i = 1:length(values)
        if ~isempty(values{1,i})
            optLL_val_P(j)= values{1,i}.optionsLikhng;
            MSE_P(j)= values{1,i}.MSE;
            IVRMSE_P(j)= values{1,i}.IVRMSE;
            j = j+1;
        end
    end
    meanoptLL_P = mean(optLL_val_P);
    meanMSE_P = mean(MSE_P);
    meanIVRMSE_P = mean(IVRMSE_P);
    save(strcat('OfoS', num2str(currentYear), '.mat'),'MSE_P', 'optLL_val_P','IVRMSE_P','meanoptLL_P', 'meanMSE_P', 'meanIVRMSE_P', '-append' );
elseif useUpdatedh0Q
    j = 1;
    for i = 1:length(values)
        if ~isempty(values{1,i})
            optLL_val_Q(j)= values{1,i}.optionsLikhng;
            MSE_Q(j)= values{1,i}.MSE;
            IVRMSE_Q(j)= values{1,i}.IVRMSE;
            j = j+1;
        end
    end
    meanoptLL_Q = mean(optLL_val_Q);
    meanMSE_Q = mean(MSE_Q);
    meanIVRMSE_Q = mean(IVRMSE_Q);
    save(strcat('OfoS', num2str(currentYear), '.mat'),'MSE_Q', 'optLL_val_Q','IVRMSE_Q','meanoptLL_Q', 'meanMSE_Q', 'meanIVRMSE_Q','-append'  );
elseif useRealVola
    j = 1;
    for i = 1:length(values)
        if ~isempty(values{1,i})
            optLL_val_RV(j)= values{1,i}.optionsLikhng;
            MSE_RV(j)= values{1,i}.MSE;
            IVRMSE_RV(j)= values{1,i}.IVRMSE;
            j = j+1;
        end
    end
    meanoptLL_RV = mean(optLL_val_RV);
    meanMSE_RV = mean(MSE_RV);
    meanIVRMSE_RV = mean(IVRMSE_RV);
    save(strcat('OfoS', num2str(currentYear), '.mat'),'MSE_RV', 'optLL_val_RV','IVRMSE_RV','meanoptLL_RV', 'meanMSE_RV', 'meanIVRMSE_RV','-append'  );
else
    j = 1;
    for i = 1:length(values)
        if ~isempty(values{1,i})
            optLL_val_est(j)= values{1,i}.optionsLikhng;
            MSE_est(j)= values{1,i}.MSE;
            IVRMSE_est(j)= values{1,i}.IVRMSE;
            j = j+1;
        end
    end
    meanoptLL_est = mean(optLL_val_est);
    meanMSE_est = mean(MSE_est);
    meanIVRMSE_est = mean(IVRMSE_est);
    save(strcat('OfoS', num2str(currentYear), '.mat'),'MSE_est', 'optLL_val_est','IVRMSE_est','meanoptLL_est', 'meanMSE_est', 'meanIVRMSE_est'  );
    
end

clearvars -except 'iii' 'row';
end
%%
clear
i = 1;
row(i,1) = meanoptLL_P;
row(i,2) = meanMSE_P;
row(i,3)=meanIVRMSE_P;

i = 2;
row(i,1) = meanoptLL_RV;
row(i,2) = meanMSE_RV;
row(i,3)=meanIVRMSE_RV;

i = 3;
row(i,1) = meanoptLL_Q;
row(i,2) = meanMSE_Q;
row(i,3)=meanIVRMSE_Q;

i = 4;
row(i,1) = meanoptLL_est;
row(i,2) = meanMSE_est;
row(i,3)=meanIVRMSE_est;
