clc; 
clearvars; 
close all;
warning('on')

path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';
year                = 2010;
useYield            = 0; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 0; % use last h_t from MLE under P as h0
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

% initialize with the data from calibrated params for each week
load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/results calibration h0calibarted esth0P OL mp verif/','params_options_',num2str(year),'_h0_calibrated_OptLL_interiorpoint_noYield.mat'));

if useRealVola || useMLEPh0
    num_params = 4;
else
    num_params = 5;
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


       
%% weekly optimization
j = 1;
uniques_weeks = unique(weeksprices);
for i = uniques_weeks(2:end-1)
%     if useRealVola
%         disp(strcat('Optimization (',goal ,') of week ',num2str(i),' in ',num2str(year),'. h_0 is not calibrated.'))
%         vola_vec = zeros(1,num_voladays);
%         vola_cell = {};
%         vola_cell{1} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
%             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j));
%         vola_cell{2} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
%             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-1);
%         vola_cell{3} = SP500_date_prices_returns_realizedvariance_interestRates(4, ...
%             SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-2);
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
%         for vola_idx = 1:num_voladays
%             if ~isempty(vola_cell{vola_idx})
%                 vola_vec(vola_idx) = vola_cell{vola_idx};
%             end
%         end
%         [~,vola_idx] =max(vola_vec>0);
%         sig2_0(i) = vola_vec(vola_idx);
%     elseif useMLEPh0
%         disp(strcat('Optimization (',goal ,') of week ',num2str(i),' in ',num2str(year),'. h_0 = h_t from MLE under P.'))
%         sig2_0(i) = sig_tmp(i);
%     else
        disp('h_0 will calibrated.')
%    end
    data_week = data(:,(weeksprices == i))';
    if isempty(data_week)
        disp(strcat('no data for week ',num2str(i),' in ',num2str(year),'!'))
        continue
    end
    struc = values{1,i};
    
    interestRates = struc.yields;
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
   
    struc = values{1,i};
    strucPrev = values{1,i-1};
    strucNew = struc;
    strucNew.hngparams = strucPrev.hngparams;
    strucNew.hngPrice      =   price_Q(strucPrev.hngparams, data_week, r_cur./252, strucPrev.sig20) ;
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
    
    disp(strucNew.MSE);
    disp(struc.MSE);
    disp(strucNew.hngparams(3));
end 
if useRealVola
    save(strcat('params_options_',num2str(year),'_h0asRealVola',num2str(num_voladays),'days_',goal,'_',algorithm,'_',txt,'.mat'),'values');
elseif useMLEPh0
    save(strcat('params_options_',num2str(year),'_h0ashtMLEP_',goal,'_',algorithm,'_',txt,'.mat'),'values');
else
    save(strcat('params_options_',num2str(year),'_h0_calibrated_','noUpd','_',txt,'.mat'),'valuesOS');
end
%for specific weeks
%save(strcat('params_Options_',num2str(year),'week2and4','_h0asRealVola_',goal,'_',algorithm,'_',txt,'.mat'),'values');
