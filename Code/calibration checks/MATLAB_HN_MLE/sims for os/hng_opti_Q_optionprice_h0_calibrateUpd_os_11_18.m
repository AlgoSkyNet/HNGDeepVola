clc; 
clearvars; 
close all;
warning('on')

path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';
datatable       = readtable('SP500_220320.csv');
dataRet            = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
years           = (dataRet(:,2)==2013);%| (data(:,2)==2012) | (data(:,2)==2013) | (data(:,2)==2014) | (data(:,2)==2015) | (data(:,2)==2016) | (data(:,2)==2017) | (data(:,2)==2018);
wednesdays     = (weekday(dataRet(:,1))==4);
doi             = years & wednesdays; %days of interest
shortdata       = dataRet(doi,:);
index           = find(doi);
thursdays       = (weekday(dataRet(:,1))==5);
doith             = years & thursdays; %days of interest
indexth           = find(doith);

yearPrev                = 2012;
formatIn                = 'dd-mmm-yyyy';

DateString_start        = strcat('01-January-',num2str(yearPrev));
DateString_end          = strcat('31-December-',num2str(yearPrev));
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
thursdays             = (weekday(date_start:date_end)==5);
Dates                   = date_start:date_end;
datesThursday = Dates(thursdays);
lastThursday = datesThursday(end);
indexLastThursday = find(dataRet(:,1)==lastThursday);
indexth = [indexLastThursday;indexth];



bound                   = [100, 100];
formatIn                = 'dd-mmm-yyyy';

% start from the first Wednesday of 2015 and finish with the last Wednesday of 2015
year = 2013;
DateString_start        = strcat('01-January-',num2str(year));
DateString_end          = strcat('31-December-',num2str(year));
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
wednessdays             = (weekday(date_start:date_end)==4);
Dates                   = date_start:date_end;
Dates                   = Dates(wednessdays);

% initialize with the data from calibrated params for each week
% load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/results calibration h0calibarted esth0P OL mp verif/','params_options_',num2str(yearPrev),'_h0_calibrated_OptLL_interiorpoint_noYield.mat'));
% strucPrev = values{1,length(values)};
% load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/results calibration h0calibarted esth0P OL mp verif/','params_options_',num2str(year),'_h0_calibrated_OptLL_interiorpoint_noYield.mat'));
load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/results calibration h0asUncondVMLEPUpdateQ esth0P OL mpoints/','params_options_',num2str(yearPrev),'_h0asUncVarUpdQ_OptLL_interiorpoint_noYield_m.mat'));
strucPrev = values{1,length(values)};
load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/results calibration h0asUncondVMLEPUpdateQ esth0P OL mpoints/','params_options_',num2str(year),'_h0asUncVarUpdQ_OptLL_interiorpoint_noYield_m.mat'));

path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);

    
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

useYield = 0;
MSE_tot = 0;
       
%% weekly optimization
j = 1;
uniques_weeks = unique(weeksprices);
for i = uniques_weeks(2:end-1)
    
    logret = dataRet(indexth(j)+1:index(j),4);
        disp('h_0 will calibrated.')

    data_week = data(:,(weeksprices == i))';
    

    if isempty(data_week)
        
        disp(strcat('no data for week ',num2str(i),' in ',num2str(year),'!'))
        continue
    end
    struc = values{1,i};
    if j > 1
        if isempty(values{1,i-1})
            strucPrev = values{1,i-2};
        else
            strucPrev = values{1,i-1};
        end
    end

    interestRates = strucPrev.yields;
    %interestRates = struc.yields;
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
        end
    end
    [~, sigma2_vals] = ll_hng_Q_n(strucPrev.hngparams(1:4), logret, shortdata(j,4)./252, strucPrev.sig20);
    sig20 = sigma2_vals(end);
    strucNew = struc;
    strucNew.hngparams = strucPrev.hngparams;
    strucNew.hngPrice      =   price_Q(strucPrev.hngparams, data_week, r_cur./252, sig20) ;
    strucNew.blsimpvhng    =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, strucNew.hngPrice');
    strucNew.epsilonhng    =   (struc.Price - strucNew.hngPrice) ./ struc.blsvega';
    s_epsilon2hng       =   mean(strucNew.epsilonhng(:).^2);
    strucNew.optionsLikhng =   -.5 * struc.numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1);
    strucNew.matr          =   [struc.Price; strucNew.hngPrice; struc.blsPrice];
    strucNew.maxAbsEr      =   max(abs(strucNew.hngPrice - struc.Price));
    strucNew.IVRMSE        =   sqrt(mean(100 * (struc.blsimpv - strucNew.blsimpvhng).^2));
    strucNew.MAPE          =   mean(abs(strucNew.hngPrice - struc.Price)./struc.Price);
    strucNew.MaxAPE        =   max(abs(strucNew.hngPrice - struc.Price)./struc.Price);
    strucNew.MSE           =   mean((strucNew.hngPrice - struc.Price).^2);
    strucNew.RMSE          =   sqrt(strucNew.MSE);
    strucNew.RMSEbls       =   sqrt(mean((struc.blsPrice - struc.Price).^2));
    valuesOS{i}           =   strucNew;    
    MSE_tot = MSE_tot + strucNew.optionsLikhng;
    disp(strucNew.MSE);
    disp(struc.MSE);
    disp(strucNew.hngparams(3));
    j = j + 1;
end 
useRealVola = 0;
useMLEPh0 = 0;
if useRealVola
    save(strcat('params_options_',num2str(year),'_h0asRealVola',num2str(num_voladays),'days_',goal,'_',algorithm,'_',txt,'.mat'),'values');
elseif useMLEPh0
    save(strcat('params_options_',num2str(year),'_h0ashtMLEP_',goal,'_',algorithm,'_',txt,'.mat'),'values');
else
    save(strcat('params_options_',num2str(year),'_h0_calibrated_','Upd.mat'),'valuesOS');
end
%for specific weeks
%save(strcat('params_Options_',num2str(year),'week2and4','_h0asRealVola_',goal,'_',algorithm,'_',txt,'.mat'),'values');
