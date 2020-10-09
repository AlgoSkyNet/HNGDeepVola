clear;
k=1;
for year=2010:2018
load(strcat('res', num2str(year), '_h0calibr_12m_avR_yield.mat'));
params = xmin_fmincon;
[fValOut1, values1]=getCalibratedDatah0(params, weeksprices, data, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index, rValue);
j = 1;
for i = 1:length(values1)
    if ~isempty(values1{1,i})
    optLL_val1(j)= values1{1,i}.optionsLikhng;
    optLL_valNorm1(j)= values1{1,i}.optionsLikNorm;
    MSE1(j)= values1{1,i}.MSE;
    IVRMSE1(j)= values1{1,i}.IVRMSE;
    j=j+1;
    end
end
row(k,1) = mean(optLL_val1);
row(k,2) = mean(MSE1);
row(k,3) = mean(IVRMSE1);
row(k,4:8)=params;
mean(optLL_valNorm1)
k = k+1;
clearvars params fValOut1 values1 optLL_val1 optLL_valNorm1 MSE1 IVRMSE1
end

clear;
load('res2011_h0RV_12m_avR_yield.mat');
params = xmin_fmincon;
[fValOut1, values1]=getCalibratedData(params, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index, rValue);
%save('resMultipleOptions2010.mat');
%values1=values;
j = 1;
for i = 1:length(values1)
    if ~isempty(values1{1,i})
    optLL_val1(j)= values1{1,i}.optionsLikhng;
    optLL_valNorm1(j)= values1{1,i}.optionsLikNorm;
    MSE1(j)= values1{1,i}.MSE;
    IVRMSE1(j)= values1{1,i}.IVRMSE;
    j=j+1;
    end
end
mean(optLL_val1)
mean(MSE1)
mean(IVRMSE1)
mean(optLL_valNorm1)
% % mean(optLL_val2)
% % 
% % mean(MSE2)
% % 
% % mean(IVRMSE2)
% % load('res2012.mat');
% % params = xmin_fmincon;
% % [fValOuth0P, values2]=getCalibratedDatah0(params, weeksprices, data,SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index);
% % 
% % for i = 1:length(values2)
% %     if ~isempty(values2{1,i})
% %     optLL_val2(i) =values2{1,i}.optionsLikhng;
% %     MSE2(i)= values2{1,i}.MSE;
% %         IVRMSE2(i)= values2{1,i}.IVRMSE;
% % end
% % end
% 
% load('res2016_h0P_6m.mat');
% params = xmin_fmincon;
% [fValOut1, values1]=getCalibratedData(params, weeksprices, data, sig_tmp(2), SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index);
% %save('resMultipleOptions2010.mat');
% for i = 1:length(values1)
%     if ~isempty(values1{1,i})
%     optLL_val1(i)= values1{1,i}.optionsLikhng;
%     MSE1(i)= values1{1,i}.MSE;
%     IVRMSE1(i)= values1{1,i}.IVRMSE;
%     end
% end
% mean(optLL_val1)
% mean(MSE1)
% mean(IVRMSE1)
% 
% % load('res2017_h0Q_new.mat');
% % params = xmin_fmincon;
% % [fValOuth0P, values2]=getCalibratedData(params, weeksprices, data, sig_tmp(indSigma), SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index);
% load('res2018_h0RV_6m.mat');
% params = xmin_fmincon;
% [fValOuth0P, values2]=getCalibratedData(params, weeksprices, data, sig_tmp, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index);
% 
% for i = 1:length(values2)
%     if ~isempty(values2{1,i})
%     optLL_val2(i) =values2{1,i}.optionsLikhng;
%     MSE2(i)= values2{1,i}.MSE;
%         IVRMSE2(i)= values2{1,i}.IVRMSE;
% end
% end
% %mean(optLL_val1)
% mean(optLL_val2)
% %mean(MSE1)
% mean(MSE2)
% %mean(IVRMSE1)
% mean(IVRMSE2)