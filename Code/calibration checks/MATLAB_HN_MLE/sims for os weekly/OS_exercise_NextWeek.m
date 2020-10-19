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
index_doi_wednesdays_corrected(j,1) = index_doi_wednesdays(i);
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
index_doi_thursdays_corrected(j,1) = index_doi_thursdays(i);
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
index_doi_fridays_corrected(j,1) = index_doi_fridays(i);
all_corrected = [index_doi_wednesdays_corrected,index_doi_thursdays_corrected,index_doi_fridays_corrected];
%indices = all_corrected(all_corrected(:,1)~=0,:);
indices = all_corrected;
indices_final = [];
for i = 2:length(indices)
    if indices(i - 1,2)~=0
        indices_final(i - 1,1:2) = [indices(i,1),indices(i - 1,2)];
    else
        indices_final(i - 1,1:2) = [indices(i,1),indices(i - 1,3)];
    end
end
%indices_final = indices_final(indices_final(:,1)~=0,:);
indices_final = [[0,0];indices_final];
indices_final_tmp = indices_final(indices_final(:,1)~=0,:);
dates_cur = dataRet(indices_final_tmp(:,1),1);
colAdd = [];
j = 1;
for i = 1:length(indices_final)
    if indices_final(i,1)
        colAdd = [colAdd; dates_cur(j)];
        j = j + 1;
    else
        colAdd = [colAdd; 0];
    end
end
indices_final = [colAdd, indices_final];
indices_final_tmp = indices_final;
indices_final = [indices_final(1:207,:);indices_final(210:end,:)];
ifUpdate = 0;
bound                   = [100, 100];

j = 1;
Dates = dataRet(index_doi_wednesdays,1);
total_struc = [];
for year_num = 2010:2018
    % h0Q=htP
    load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/calibration h0MLEPht/OptLL/','params_options_',num2str(year_num),'_h0ashtMLEP_OptLL_interiorpoint_noYield_m.mat'));
    load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year_num),'_mle_opt_h0est.mat'));
    
    % h0Q=uncVar
%     load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P for Update/','weekly_',num2str(year_num),'_mle_opt_h0est_UpdateQ.mat'));
%     load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/calibration h0UncVar/withUpdate1year/OptLL/','params_options_',num2str(year_num),'_h0asUncVarUpdQ_OptLL_interiorpoint_noYield_m.mat'));
    
    % h0Q calibrated
%     load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year_num),'_mle_opt_h0est.mat'));
%     load(strcat('/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/calibration h0calibrated/OptLL/results calibr h0calibarted esth0P OL mp verif/','params_options_',num2str(year_num),'_h0_calibrated_OptLL_interiorpoint_noYield.mat'));
    
    
    num_weeks = length(values);
    if isempty(values{1,1})
         for k = 2:num_weeks
%              if isempty(values{1,k})
%                     indices_final(j)=0;
%                 end
            if isempty(values{1,k})
                c=3;
            end
                total_struc{1,j} = values{1,k};
            if ~isempty(values{1,k})
                
                c=3;
                total_struc{1,j}.params_P = params_tmp_P(k,:);
                total_struc{1,j}.params_Q = params_tmp(k,:);
                total_struc{1,j}.vola_tmp = vola_tmp(k);
            end
                
                    j = j + 1;
            %end
            end
    else
        for k = 1:num_weeks
%              if isempty(values{1,k})
%                     indices_final(j)=0;
%                 end
            total_struc{1,j} = values{1,k};
            if ~isempty(values{1,k})
                %total_struc{1,j} = values{1,k};
                c=3;
                total_struc{1,j}.params_P = params_tmp_P(k,:);
                total_struc{1,j}.params_Q = params_tmp(k,:);
                total_struc{1,j}.vola_tmp = vola_tmp(k);
            end
         %if ~isempty(values{1,k})
              
                
                    j = j + 1;
           % end
        
        end
    end
   
end

% bounds for maturity, moneyness, volumes, interest rates
Type                    = 'call';
MinimumVolume           = 100;
MinimumOpenInterest     = 100;
IfCleanNans             = 1;
TimeToMaturityInterval  = [8, 250];
MoneynessInterval       = [0.9, 1.1];
useYield = 0;
%indices_final = indices_final(indices_final(:,1)~=0,:);


cur_week = 1;
struct_ind = 2;
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
        uniques_weeks = uniques_weeks(2:end);
        cur_week = uniques_weeks(1);
    end
    flag = 0;
    i = uniques_weeks(1);
    if Dates(struct_ind)~=indices_final(struct_ind,1)
        a = 9;
    end
    valuesOS=[];
    while i <= max(uniques_weeks)
        if i == max(uniques_weeks)
             c=3;
        end

        if flag
            i = i + 2;
            %struct_ind = struct_ind + 1;
            flag = 0;
        end
        if i>max(uniques_weeks)
                continue;
                
            end

        data_week = data(:,(weeksprices == i))';
        if isempty(data_week) || ~indices_final(struct_ind,2)
            struct_ind = struct_ind +2;
            cur_week = cur_week + 1;
            flag = 1;
            continue;
        end
        
        logret = dataRet(indices_final(struct_ind,3):indices_final(struct_ind,2),4);
        
        strucCur = total_struc{1, struct_ind};
        strucPrev = total_struc{1,struct_ind-1};
        interestRates = strucCur.yields;
        intCheck = SP500_date_prices_returns_realizedvariance_interestRates(5:9,SP500_date_prices_returns_realizedvariance_interestRates(1,:)==dataRet(index_doi_wednesdays(cur_week - 1),1));
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
        strucOSCur                  = struct();
%        params_for_update   = strucPrev.params_Q(1:4);        
        strucOSCur.hngparams   = strucPrev.hngparams;    

%         [~, sigma2_vals] = ll_hng_Q_n(params_for_update, logret, r, strucPrev.sig20);
%         sig20 = sigma2_vals(end);
        sig20 = strucPrev.sig20;
        
        strucOSCur.hngPrice         = price_Q(strucOSCur.hngparams, data_week, r_cur./252, sig20) ;
        strucOSCur.blsimpvhng       = blsimpv(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, strucOSCur.hngPrice');
        strucOSCur.epsilonhng       = (strucCur.Price - strucOSCur.hngPrice) ./ strucCur.blsvega';
        s_epsilon2hng               = mean(strucOSCur.epsilonhng(:).^2);
        strucOSCur.optionsLikhng    = -.5 * strucCur.numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1 + sum(log(strucCur.blsvega)) * 2/strucCur.numOptions);
        strucOSCur.optionsLikhngNorm    = -.5 * (log(2 * pi) + log(s_epsilon2hng) + 1 + sum(log(strucCur.blsvega)) * 2/strucCur.numOptions);

        strucOSCur.matr             = [strucCur.Price; strucOSCur.hngPrice; strucCur.blsPrice];
        strucOSCur.maxAbsEr         = max(abs(strucOSCur.hngPrice - strucCur.Price));
        strucOSCur.IVRMSE           = sqrt(mean(100 * (strucCur.blsimpv - strucOSCur.blsimpvhng).^2));
        strucOSCur.MAPE             = mean(abs(strucOSCur.hngPrice - strucCur.Price)./strucCur.Price);
        strucOSCur.MaxAPE           = max(abs(strucOSCur.hngPrice - strucCur.Price)./strucCur.Price);
        strucOSCur.MSE              = mean((strucOSCur.hngPrice - strucCur.Price).^2);
        strucOSCur.RMSE             = sqrt(strucOSCur.MSE);
        strucOSCur.RMSEbls          = sqrt(mean((strucCur.blsPrice - strucCur.Price).^2));
        strucOSCur.sig20 = sig20;
        strucOSCur.sig20BeforeUpdate = strucPrev.sig20;
        strucOSCur.logret = logret;
        parNum = 4;
        strucOSCur.AIC = 2*parNum - strucOSCur.optionsLikhng;
        strucOSCur.BIC = parNum * log(strucCur.numOptions) - 2*strucOSCur.optionsLikhng;
        strucOSCur.AICc = strucOSCur.AIC + (2*parNum^2+2*parNum)/(strucCur.numOptions - parNum - 1);

        valuesOS{i}                 = strucOSCur;
                
        disp(strucOSCur.MSE);
        disp(strucCur.MSE);
        cur_week = cur_week + 1;
        struct_ind = struct_ind + 1;
        i = i + 1;
    end
    % h0Q=htP
    save(strcat('params_options_',num2str(year_num),'_h0_MLEP_','noUpd.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_MLEP_','Upd.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_MLEP_','UpdWithMLEPparams.mat'),'valuesOS');
    % h0Q=uncVar
    %save(strcat('params_options_',num2str(year_num),'_h0_uncVarUpdQ1year_','noUpd.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_uncVarUpdQ1year_','Upd.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_uncVarUpdQ1year_UpdWithMLEPparams.mat'),'valuesOS');
    % h0Q=calibrated
    %save(strcat('params_options_',num2str(year_num),'_h0_calibr_','Upd.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_calibr_','UpdWithMLEPparams.mat'),'valuesOS');
    %save(strcat('params_options_',num2str(year_num),'_h0_calibr_','noUpd.mat'),'valuesOS');
end

