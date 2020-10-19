clc;
clearvars;
close all;
warning('on')

path                =  'C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/Data/Datasets';
% path_r       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
load('SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');

stock_ind           = 'SP500';
datatable           = readtable('SP500_220320.csv');
dataRet             = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
years               = (dataRet(:,2)==2011);
load('res2010.mat','values');
%load('res2010_h0Q.mat','values');

hngparams = values{1,length(values)}.hngparams(1:4);
sig20 = values{1,length(values)}.sig20;
wednesdays          = weekday(dataRet(:,1))==4;
doi_wednesdays      = years & wednesdays; %days of interest

data_doi_wednesdays = dataRet(doi_wednesdays,:);

index_doi_wednesdays= find(doi_wednesdays);
% 
% index_doi_wednesdays_corrected = [];
% j = 1;
% for i = 2:length(index_doi_wednesdays)
%     if index_doi_wednesdays(i) - index_doi_wednesdays(i - 1) <= 5
%         index_doi_wednesdays_corrected(j,1) = index_doi_wednesdays(i - 1);
%     elseif (index_doi_wednesdays(i) - index_doi_wednesdays(i - 1) > 5) && ((index_doi_wednesdays(i) - index_doi_wednesdays(i - 1)) <= 9)
%         index_doi_wednesdays_corrected(j,1) = index_doi_wednesdays(i - 1);
%         j = j + 1;
%         index_doi_wednesdays_corrected(j,1) = 0;
%         
%         
%     else
%         index_doi_wednesdays_corrected(j,1) = index_doi_wednesdays(i - 1);
%         j = j + 1;
%         index_doi_wednesdays_corrected(j,1) = 0;
%         j = j + 1;
%         index_doi_wednesdays_corrected(j,1) = 0;
%        
%         
%         
%     end
%     j = j + 1;
% end
% 
% all_corrected = [index_doi_wednesdays_corrected];
indices_final = index_doi_wednesdays;

Dates = data_doi_wednesdays(:, 1);
ifUpdate = 0;
bound                   = [100, 100];

% j = 1;
% total_struc = [];
% for year_num = 2011
%     %load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/results calibration h0asUncondVMLEPUpdateQ esth0P OL mpoints/','params_options_',num2str(year_num),'_h0asUncVarUpdQ_OptLL_interiorpoint_noYield_m.mat'));
%     %load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/results calibration hoashtMLEP esthoP OL mpoints/','params_options_',num2str(year_num),'_h0ashtMLEP_OptLL_interiorpoint_noYield_m.mat'));
%     load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/results 12m/res',num2str(year_num),'.mat'));
%     num_weeks = length(values);
%     if isempty(values{1,1})
%         for k = 2:num_weeks
%             total_struc{1,j} = values{1,k};
%             j = j + 1;
% 
%         end
%     else
%         for k = 1:num_weeks
%         total_struc{1,j} = values{1,k};
%         j = j + 1;
%         
%         end
%     end
%    
% end

% bounds for maturity, moneyness, volumes, interest rates
Type                    = 'call';
MinimumVolume           = 100;
MinimumOpenInterest     = 100;
IfCleanNans             = 1;
TimeToMaturityInterval  = [8, 250];
MoneynessInterval       = [0.9, 1.1];
useYield = 0;
load('params_options_2011_h0_calibrated_OptLL_interiorpoint_noYield');
cur_week = 1;
struct_ind = 1;
MSE_tot = 0;
for year_num = 2011
    path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year_num), '.mat');
    load(path_);
    
    [OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptions(Dates, Type, ...
        TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans,...
        TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume, ...
        OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
        TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);
    weeksprices = week(datetime([OptionsStruct.date], 'ConvertFrom', 'datenum'));
    uniques_weeks = unique(weeksprices);
    idxj  = 1:length(uniques_weeks);
    data = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega; OptionsStruct.implied_volatility];
    
    flag = 0;
    i = uniques_weeks(1);
    valuesOS=[];
    struct_ind = i;
    while i <= max(uniques_weeks)
        
        
        strucCur =values{1,struct_ind};
        data_week = data(:,(weeksprices == i))';
        if isempty(data_week) || isempty(strucCur)
            struct_ind = struct_ind + 1;
            cur_week = cur_week + 1;
            flag = 1;
            i = i + 1;
            continue;
        end
        
       
        
        indNaN = find(isnan(strucCur.blsimpv));
    strucCur.num_NaN_implVols = length(indNaN);
    strucCur.blsimpv(indNaN) = data_week(indNaN, 6);
    interestRates = strucCur.yields;
        r_cur = zeros(length(data_week), 1);
        for k = 1:length(data_week)
            if data_week(k, 2) < 21 && ~isnan(interestRates(1))
                r_cur(k) = interestRates(1);
            else
                notNaN = ~isnan(interestRates);
                daylengths = [21, 42, 13*5, 126, 252]./252;
                r_cur(k) = interp1(daylengths(notNaN), interestRates(notNaN), data_week(k, 2)./252);
            end
        end
       
    strucCur.blsvega = blsvega(data_week(:, 4),  data_week(:, 3), r_cur(:), data_week(:, 2)/252, strucCur.blsimpv(:));
        interestRates = [0.00115471698113208,NaN,0.00135660377358491,0.00191886792452830,0.00298679245283019];
        r_cur = zeros(length(data_week), 1);
        for k = 1:length(data_week)
            if data_week(k, 2) < 21 && ~isnan(interestRates(1))
                r_cur(k) = interestRates(1);
            else
                notNaN = ~isnan(interestRates);
                daylengths = [21, 42, 13*5, 126, 252]./252;
                r_cur(k) = interp1(daylengths(notNaN), interestRates(notNaN), data_week(k, 2)./252);
            end
        end
        r = interestRates(end)/252;
        %[~, sigma2_vals] = ll_hng_Q_n(strucPrev.hngparams(1:4), logret, r, strucPrev.sig20);
        %sig20 = strucPrev.sig20;%sigma2_vals(end);
        
        strucOSCur                  = struct();
        strucOSCur.hngparams        = hngparams;
        strucOSCur.hngPrice         = price_Q(strucOSCur.hngparams, data_week, r_cur./252, sig20) ;
        strucOSCur.blsimpvhng       = blsimpv(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, strucOSCur.hngPrice');
       

        strucOSCur.epsilonhng       = (strucCur.Price - strucOSCur.hngPrice) ./ strucCur.blsvega';
        s_epsilon2hng               = mean(strucOSCur.epsilonhng(:).^2);
        strucOSCur.optionsLikhng    = -.5 * strucCur.numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1);
        strucOSCur.matr             = [strucCur.Price; strucOSCur.hngPrice; strucCur.blsPrice];
        strucOSCur.maxAbsEr         = max(abs(strucOSCur.hngPrice - strucCur.Price));
        strucOSCur.IVRMSE           = sqrt(mean(100 * (strucCur.blsimpv - strucOSCur.blsimpvhng).^2));
        strucOSCur.MAPE             = mean(abs(strucOSCur.hngPrice - strucCur.Price)./strucCur.Price);
        strucOSCur.MaxAPE           = max(abs(strucOSCur.hngPrice - strucCur.Price)./strucCur.Price);
        strucOSCur.MSE              = mean((strucOSCur.hngPrice - strucCur.Price).^2);
        strucOSCur.RMSE             = sqrt(strucOSCur.MSE);
        strucOSCur.RMSEbls          = sqrt(mean((strucCur.blsPrice - strucCur.Price).^2));
        strucOSCur.sig20 = sig20;
        %strucOSCur.sig20BeforeUpdate = strucPrev.sig20;
        %strucOSCur.logret = logret;
        valuesOS{i}                 = strucOSCur;
                
        disp(strucOSCur.MSE);
        %disp(strucCur.MSE);
        cur_week = cur_week + 1;
        struct_ind = struct_ind + 1;
        i = i + 1;
    end
    %save(strcat('params_options_',num2str(year_num),'_h0_MLEP_','Upd.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_calibr_','Upd.mat'),'valuesOS');
    save(strcat('params_Multoptions_',num2str(year_num),'_h0_calibr_','noUpd.mat'),'valuesOS');
end

