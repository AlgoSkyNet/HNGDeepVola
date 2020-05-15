clc;
clearvars;
close all;
warning('on')

path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
path_r       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
load(path_r);

stock_ind           = 'SP500';
datatable           = readtable('SP500_220320.csv');
dataRet             = [datenum(datatable.Date),year(datatable.Date),datatable.AdjClose,[0;log(datatable.AdjClose(2:end))-log(datatable.AdjClose(1:end-1))]];
years               = (dataRet(:,2)==2010) | (dataRet(:,2)==2011) ...
    | (dataRet(:,2)==2012) | (dataRet(:,2)==2013) ...
    | (dataRet(:,2)==2014) | (dataRet(:,2)==2015) ...
    | (dataRet(:,2)==2016) | (dataRet(:,2)==2017) | (dataRet(:,2)==2018);
wednesdays          = weekday(dataRet(:,1))==4;
thursdays           = weekday(dataRet(:,1))==5;
fridays             = weekday(dataRet(:,1))==6;
doi_wednesdays      = years & wednesdays; %days of interest
doi_thursdays       = years & thursdays; %days of interest
doi_fridays         = years & fridays; %days of interest

data_doi_wednesdays = dataRet(doi_wednesdays,:);
data_doi_thursdays  = dataRet(doi_thursdays,:);
data_doi_fridays  = dataRet(doi_fridays,:);

index_doi_wednesdays= find(doi_wednesdays);
index_doi_thursdays = find(doi_thursdays);
index_doi_fridays = find(doi_fridays);

index_doi_wednesdays_corrected = [];
j = 1;
for i = 2:length(index_doi_wednesdays)
    if index_doi_wednesdays(i) - index_doi_wednesdays(i - 1) <= 5
        index_doi_wednesdays_corrected(j,1) = index_doi_wednesdays(i - 1);
    elseif (index_doi_wednesdays(i) - index_doi_wednesdays(i - 1) > 5) && ((index_doi_wednesdays(i) - index_doi_wednesdays(i - 1)) <= 9)
        index_doi_wednesdays_corrected(j,1) = index_doi_wednesdays(i - 1);
        j = j + 1;
        index_doi_wednesdays_corrected(j,1) = 0;
        
        
    else
        index_doi_wednesdays_corrected(j,1) = index_doi_wednesdays(i - 1);
        j = j + 1;
        index_doi_wednesdays_corrected(j,1) = 0;
        j = j + 1;
        index_doi_wednesdays_corrected(j,1) = 0;
        
        
        
    end
    j = j + 1;
end

index_doi_thursdays_corrected = [];
j = 1;
for i = 2:length(index_doi_thursdays)
    if index_doi_thursdays(i) - index_doi_thursdays(i - 1) <= 5
        index_doi_thursdays_corrected(j,1) = index_doi_thursdays(i - 1);
    elseif (index_doi_thursdays(i) - index_doi_thursdays(i - 1) > 5) && (index_doi_thursdays(i) - index_doi_thursdays(i - 1)) <= 9
        index_doi_thursdays_corrected(j,1) = index_doi_thursdays(i - 1);
        j = j + 1;
        index_doi_thursdays_corrected(j,1) = 0;
        
        
    else
        index_doi_thursdays_corrected(j,1) = index_doi_thursdays(i - 1);
        j = j + 1;
        index_doi_thursdays_corrected(j,1) = 0;
        j = j + 1;
        index_doi_thursdays_corrected(j,1) = 0;
        
        
        
    end
    j = j + 1;
end

index_doi_fridays_corrected = [];
j = 1;
for i = 2:length(index_doi_fridays)
    if index_doi_fridays(i) - index_doi_fridays(i - 1) <= 5
        index_doi_fridays_corrected(j,1) = index_doi_fridays(i - 1);
    elseif (index_doi_fridays(i) - index_doi_fridays(i - 1) > 5) && (index_doi_fridays(i) - index_doi_fridays(i - 1)) <= 9
        index_doi_fridays_corrected(j,1) = index_doi_fridays(i - 1);
        j = j + 1;
        index_doi_fridays_corrected(j,1) = 0;
        
        
    else
        index_doi_fridays_corrected(j,1) = index_doi_fridays(i - 1);
        j = j + 1;
        index_doi_fridays_corrected(j,1) = 0;
        j = j + 1;
        index_doi_fridays_corrected(j,1) = 0;
    end
    j = j + 1;
end
all_corrected = [index_doi_wednesdays_corrected,index_doi_thursdays_corrected,index_doi_fridays_corrected];
%indices = all_corrected(all_corrected(:,1)~=0,:);
indices = all_corrected;
indices_final = [];
j= 1;
for i = 1:length(indices)
    %if indices(i,1)~=0
    indices_final(j,1) = indices(i,1);
    if indices(i,2)~=0
        indices_final(j,2) = indices(i,2);
    elseif indices(i,3)~=0
        indices_final(j,2) = indices(i,3);
    else
        a=0;
    end
    j = j+1;
    %end
    
end
%
indices_final = [indices_final;[index_doi_wednesdays(end),index_doi_thursdays(end)]];
%indices_final = indices_final(indices_final(:,1)~=0,:);

ifUpdate = 0;
bound                   = [100, 100];

j = 1;

jcur = 1;
total_struc = [];
num_weeks_total = 0;
for year_num = 2010:2018
    
    % h0Q=htP
        load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/calibration h0MLEPht/OptLL/','params_options_',num2str(year_num),'_h0ashtMLEP_OptLL_interiorpoint_noYield_m.mat'));
        load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year_num),'_mle_opt_h0est.mat'));
    
    % h0Q=uncVar
%         load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P for Update/','weekly_',num2str(year_num),'_mle_opt_h0est_UpdateQ.mat'));
%         load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/calibration h0UncVar/withUpdate1year/OptLL/','params_options_',num2str(year_num),'_h0asUncVarUpdQ_OptLL_interiorpoint_noYield_m.mat'));
    
    % h0Q calibrated
%     load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year_num),'_mle_opt_h0est.mat'));
%     load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/calibration h0calibrated/OptLL/results calibr h0calibarted esth0P OL mp verif/','params_options_',num2str(year_num),'_h0_calibrated_OptLL_interiorpoint_noYield.mat'));
    
    
    
    
    num_weeks = length(values);
    
    
    if isempty(values{1,1})
        for k = 2:num_weeks
            
            if ~isempty(values{1,k})
                total_struc{1,j} = values{1,k};
                total_struc{1,j}.params_P = params_tmp_P(k,:);
                total_struc{1,j}.params_Q = params_tmp(k,:);
                total_struc{1,j}.vola_tmp = vola_tmp(k);
                
                    j = j + 1;
            end
            if isempty(values{1,k})
                    indices_final(j)=0;
                end
            
            
            
            if ~isempty(values{1,k})
                num_weeks_total = num_weeks_total + 1;
            end
            jcur = jcur + 1;
        end
    else
        for k = 1:num_weeks
            
            if isempty(values{1,k})
                    indices_final(j)=0;
                end
            
            if ~isempty(values{1,k})
                total_struc{1,j} = values{1,k};
                total_struc{1,j}.params_P = params_tmp_P(k,:);
                total_struc{1,j}.params_Q = params_tmp(k,:);
                total_struc{1,j}.vola_tmp = vola_tmp(k);
                if vola_tmp(k)==0
                    b=2;
                end
                    j = j + 1;
            else
                h=0;
            end
            if ~isempty(values{1,k})
                num_weeks_total = num_weeks_total + 1;
            end
            jcur = jcur + 1;
        end
        
    end
    
    
end
% bounds for maturity, moneyness, volumes, interest rates
indices_final = indices_final(indices_final(:,1)~=0,:);
Dates = dataRet(indices_final(:,2),1);

Type                    = 'call';
MinimumVolume           = 100;
MinimumOpenInterest     = 100;
IfCleanNans             = 1;
TimeToMaturityInterval  = [8, 250];
MoneynessInterval       = [0.9, 1.1];
useYield = 0;

cur_week = 1;
struct_ind = 1;
MSE_tot = 0;
for year_num = 2010:2018
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
    if year_num == 2010
        uniques_weeks = uniques_weeks(1:end);
        cur_week = uniques_weeks(1);
    end
    flag = 0;
    i = uniques_weeks(1);
    valuesOS=[];
    
    for i = (uniques_weeks)
        if i == max(uniques_weeks)
             c=3;
        end
        if flag
            i = i + 1;
            flag = 0;
        end
        if i>max(uniques_weeks)
            continue;
            
        end
        
        data_week = data(:,(weeksprices == i))';
        if isempty(data_week) || ~indices_final(struct_ind,1)
            struct_ind = struct_ind + 1;
            cur_week = cur_week + 1;
            flag = 1;
            continue;
        end
        
        logret = dataRet(indices_final(struct_ind,2),4);
        
        strucCur = total_struc{1, struct_ind};
        interestRates = strucCur.yields;
        intCheck = SP500_date_prices_returns_realizedvariance_interestRates(5:9,SP500_date_prices_returns_realizedvariance_interestRates(1,:)==dataRet(indices_final(struct_ind,1),1));
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
        if struct_ind == 3
            c=5;
        end
%         params_for_update   = strucCur.params_Q(1:4);
%         [~, sigma2_vals] = ll_hng_Q_n(params_for_update, logret, r, strucCur.sig20);
%         sig20 = sigma2_vals(end);
        
        sig20 = strucCur.sig20;
        
        sig20BeforeUpdate = strucCur.sig20;
        strucOSCur                  = struct();
        strucOSCur.hngparams        = strucCur.hngparams;
        strucOSCur.Price         =   data_week(:, 1)';
        strucOSCur.numOptions = length(strucOSCur.Price);
        strucOSCur.blsPrice      =   blsprice(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, strucCur.vola_tmp, 0)';
        strucOSCur.blsimpv       =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, data_week(:, 1));
        indNaN = find(isnan(strucOSCur.blsimpv));
        strucOSCur.num_NaN_implVols = length(indNaN);
        strucOSCur.blsimpv(indNaN) = data_week(indNaN, 6);
        strucOSCur.blsvega = blsvega(data_week(:, 4),  data_week(:, 3), r_cur(:), data_week(:, 2)/252, strucOSCur.blsimpv(:));
        
        strucOSCur.hngPrice         = price_Q(strucOSCur.hngparams, data_week, r_cur./252, sig20) ;
        strucOSCur.blsimpvhng       = blsimpv(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, strucOSCur.hngPrice');
        strucOSCur.epsilonhng       = (strucOSCur.Price - strucOSCur.hngPrice) ./ strucOSCur.blsvega';
        s_epsilon2hng               = mean(strucOSCur.epsilonhng(:).^2);
        strucOSCur.optionsLikhng    = -.5 * strucOSCur.numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1 + sum(log(strucOSCur.blsvega)) * 2/strucOSCur.numOptions);
        strucOSCur.optionsLikhngNorm    = -.5 * (log(2 * pi) + log(s_epsilon2hng) + 1 + sum(log(strucOSCur.blsvega)) * 2/strucOSCur.numOptions);
        strucOSCur.matr             = [strucOSCur.Price; strucOSCur.hngPrice; strucOSCur.blsPrice];
        strucOSCur.maxAbsEr         = max(abs(strucOSCur.hngPrice - strucOSCur.Price));
        strucOSCur.IVRMSE           = sqrt(mean(100 * (strucOSCur.blsimpv - strucOSCur.blsimpvhng).^2));
        strucOSCur.MAPE             = mean(abs(strucOSCur.hngPrice - strucOSCur.Price)./strucOSCur.Price);
        strucOSCur.MaxAPE           = max(abs(strucOSCur.hngPrice - strucOSCur.Price)./strucOSCur.Price);
        strucOSCur.MSE              = mean((strucOSCur.hngPrice - strucOSCur.Price).^2);
        strucOSCur.RMSE             = sqrt(strucOSCur.MSE);
        strucOSCur.RMSEbls          = sqrt(mean((strucOSCur.blsPrice - strucOSCur.Price).^2));
        strucOSCur.sig20 = sig20;
        strucOSCur.sig20BeforeUpdate = sig20BeforeUpdate;
        strucOSCur.logret = logret;
        parNum = 4;
        strucOSCur.AIC = 2*parNum - strucOSCur.optionsLikhng;
        strucOSCur.BIC = parNum * log(strucOSCur.numOptions) - 2*strucOSCur.optionsLikhng;
        strucOSCur.AICc = strucOSCur.AIC + (2*parNum^2+2*parNum)/(strucOSCur.numOptions - parNum - 1);
        
        
        
        valuesOS{i}                 = strucOSCur;
        
        disp(strucOSCur.MSE);
        disp(strucCur.MSE);
        cur_week = cur_week + 1;
        struct_ind = struct_ind + 1;
        i = i + 1;
    end
    
    % h0Q=htP
    save(strcat('params_options_',num2str(year_num),'_h0_MLEP_','noUpd_nextday.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_MLEP_','Upd_nextday.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_MLEP_','UpdWithMLEPparams_nextday.mat'),'valuesOS');
    
    % h0Q=uncVar
    %save(strcat('params_options_',num2str(year_num),'_h0_uncVarUpdQ1year_','noUpd_nextday.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_uncVarUpdQ1year_','Upd_nextday.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_uncVarUpdQ1year_UpdWithMLEPparams_nextday.mat'),'valuesOS');
    
    % h0Q=calibrated
    %save(strcat('params_options_',num2str(year_num),'_h0_calibr_','Upd_nextday.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_calibr_','UpdWithMLEPparams_nextday.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_calibr_','noUpd_nextday.mat'),'valuesOS');
    
    
end

