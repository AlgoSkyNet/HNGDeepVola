clear;
k = 1;
ifHalfYear = 0;
if ifHalfYear
    %periodstr = '_6m_avR_yield';
    periodstr = '_6m';
else
    %periodstr = '_12m_avR_yield';
    periodstr = '';
end





row = zeros(36,9);
for year = 2010:2018
    load(strcat('res', num2str(year), '_h0P', periodstr));
    params = xmin_fmincon;
    indSigma = find(sig_tmp);
    indSigma = indSigma(1);
    %[~, values1] = getCalibratedData(params, weeksprices, data, sig_tmp(indSigma), SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index, rValue);
    [~, values1] = getCalibratedData(params, weeksprices, data, sig_tmp(indSigma), SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index);
    j = 1;
    for i = 1:length(values1)
        if ~isempty(values1{1,i})
            optLL_val(j) = values1{1,i}.optionsLikhng;
            optLL_valNorm(j) = values1{1,i}.optionsLikNorm;
            MSE(j) = values1{1,i}.MSE;
            IVRMSE(j) = values1{1,i}.IVRMSE;
            j = j+1;
        end
    end
    row(k, 1:4) = params(:)';
    row(k, 5) = sig_tmp(indSigma);
    row(k, 6) = mean(optLL_val);
    row(k, 7) = mean(optLL_valNorm);
    row(k, 8) = mean(MSE);
    row(k, 9) = mean(IVRMSE);
    
    clearvars params values1 optLL_val optLL_valNorm MSE IVRMSE
    
    k = k + 1;

    load(strcat('res', num2str(year), '_h0RV', periodstr));
    params = xmin_fmincon;
    %[~, values1]=getCalibratedData(params, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index, rValue);
    [~, values1]=getCalibratedData(params, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index);
    j = 1;
    for i = 1:length(values1)
        if ~isempty(values1{1,i})
            optLL_val(j)= values1{1,i}.optionsLikhng;
            optLL_valNorm(j)= values1{1,i}.optionsLikNorm;
            MSE(j)= values1{1,i}.MSE;
            IVRMSE(j)= values1{1,i}.IVRMSE;
            j = j+1;
        end
    end
    row(k, 1:4) = params(:)';
    row(k, 5) = sig_tmp;
    row(k, 6) = mean(optLL_val);
    row(k, 7) = mean(optLL_valNorm);
    row(k, 8) = mean(MSE);
    row(k, 9) = mean(IVRMSE);
    
    clearvars params values1 optLL_val optLL_valNorm MSE IVRMSE
    
    k = k + 1;
    
    load(strcat('res', num2str(year), '_h0Q', periodstr));
    params = xmin_fmincon;
    indSigma = find(sig_tmp);
    indSigma = indSigma(1);
    %[~, values1] = getCalibratedData(params, weeksprices, data, sig_tmp(indSigma), SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index, rValue);
    [~, values1] = getCalibratedData(params, weeksprices, data, sig_tmp(indSigma), SP500_date_prices_returns_realizedvariance_interestRates, Dates, dataRet, vola_tmp, index);
    j = 1;
    for i = 1:length(values1)
        if ~isempty(values1{1,i})
            optLL_val(j) = values1{1,i}.optionsLikhng;
            optLL_valNorm(j) = values1{1,i}.optionsLikNorm;
            MSE(j) = values1{1,i}.MSE;
            IVRMSE(j) = values1{1,i}.IVRMSE;
            j = j+1;
        end
    end
    row(k, 1:4) = params(:)';
    row(k, 5) = sig_tmp(indSigma);
    row(k, 6) = mean(optLL_val);
    row(k, 7) = mean(optLL_valNorm);
    row(k, 8) = mean(MSE);
    row(k, 9) = mean(IVRMSE);
    
    clearvars params values1 optLL_val optLL_valNorm MSE IVRMSE
    
    k = k+1;
    
    %load(strcat('res', num2str(year), '_h0calibr', periodstr));
    load(strcat('res', num2str(year), periodstr));
    params = xmin_fmincon;
    %[~, values1]=getCalibratedDatah0(params, weeksprices, data, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index, rValue);
    [~, values1]=getCalibratedDatah0(params, weeksprices, data, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index);
    j = 1;
    for i = 1:length(values1)
        if ~isempty(values1{1,i})
            optLL_val(j)= values1{1,i}.optionsLikhng;
            optLL_valNorm(j)= values1{1,i}.optionsLikNorm;
            MSE(j)= values1{1,i}.MSE;
            IVRMSE(j)= values1{1,i}.IVRMSE;
            j=j+1;
        end
    end
    row(k, 1:5) = params(:)';
    row(k, 6) = mean(optLL_val);
    row(k, 7) = mean(optLL_valNorm);
    row(k, 8) = mean(MSE);
    row(k, 9) = mean(IVRMSE);
    
    
    
    clearvars params values1 optLL_val optLL_valNorm MSE IVRMSE
    
    k = k+1;
end

%save(strcat('data_MO_IS', periodstr, '_rAvYieldMLEP_rAvYieldCalibr_10_18.mat'),'row');
save(strcat('data_MO_IS', periodstr, '_rWeekTbillMLEP_rWeekTbillCalibr_10_18.mat'),'row');

