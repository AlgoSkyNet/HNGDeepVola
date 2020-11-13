clc; clearvars;
load("data_for_IVfullnormal_intrinsic.mat")
N = length(param);
S0 = 2000;
Moneyness = 1.1:-0.025:0.9;
Strikes = S0./Moneyness;
Maturities = [10:30:250];
imp_vola_price = zeros(N,9,9);
imp_vola_forecast = zeros(N,9,9);
for i =1:N
    if mod(i,100)==0,disp(i);end
    for m =1:9
        for s=1:9
            imp_vola_price(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,price(i,m,s));
            imp_vola_forecast(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,forecast(i,m,s));
        end
    end
end
ivrmse_approx = (price-forecast).^2./vega.^2;
ivrmse = (imp_vola_price-imp_vola_forecast).^2;
diff = reshape(nanmean(abs(ivrmse(1:N,:,:)-ivrmse_approx(1:N,:,:)),1),9,9);
rel_diff_mse = reshape(nanmean(abs(ivrmse(1:N,:,:)-ivrmse_approx(1:N,:,:))./ivrmse(1:N,:,:),1),9,9);
rel_diff_mse = reshape(nanmean(abs(ivrmse(1:N,:,:)-ivrmse_approx(1:N,:,:))./ivrmse(1:N,:,:),1),9,9);

nancounter = reshape(sum(isnan(ivrmse),1),9,9);

mape_pw = abs(ivrmse(1:N,:,:)-ivrmse_approx(1:N,:,:))./ivrmse(1:N,:,:);
mean_mape = reshape(mean(abs(imp_vola_price-imp_vola_forecast)./imp_vola_price,1),9,9);
figure('Name','Vega vs differce')
scatter(reshape(vega,81*N,1),reshape(mape_pw,81*N,1))
set(gca, 'XScale', 'log'),set(gca, 'YScale', 'log')

% 
% %%
% N=500
% imp_vola_price2 = zeros(N,9,9);
% imp_vola_forecast2 = zeros(N,9,9);
% for i =1:N
%     if mod(i,100)==0,disp(i);end
%     for m =1:9
%         for s=1:9
%             imp_vola_price2(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,price(i,m,s),"Method","search","Tolerance",0.01);
%             imp_vola_forecast2(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,forecast(i,m,s),"Method","search","Tolerance",0.01);
%         end
%     end
% end
% ivrmse2 = (imp_vola_price2-imp_vola_forecast2).^2;
% diff2 = reshape(nanmean(abs(ivrmse2(1:N,:,:)-ivrmse_approx(1:N,:,:)),1),9,9);
% rel_diff2 = reshape(nanmean(abs(ivrmse2(1:N,:,:)-ivrmse_approx(1:N,:,:))./ivrmse2(1:N,:,:),1),9,9);
% nancounter2 = reshape(sum(isnan(ivrmse2),1),9,9);
% nanmean2 = reshape(mean(isnan(ivrmse2),1),9,9);
% figure('Name','Vega vs differce 2')
% mape_pw2 = abs(ivrmse2(1:N,:,:)-ivrmse_approx(1:N,:,:))./ivrmse2(1:N,:,:);
% %scatter(reshape(vega,81*N,1),reshape(mape_pw2,81*N,1))
% %set(gca, 'XScale', 'log'),set(gca, 'YScale', 'log')
% 
% % %%
% N = length(param);
% imp_vola_price3 = zeros(N,9,9);
% imp_vola_forecast3 = zeros(N,9,9);
% for i =1:N
%     if mod(i,100)==0,disp(i);end
%     for m =1:9
%         for s=1:9
%             imp_vola_price3(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,price(i,m,s));
%             imp_vola_forecast3(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,0.5*(price(i,m,s)+forecast(i,m,s)));
%         end
%     end
% end
% ivrmse3 = (imp_vola_price3-imp_vola_forecast3).^2;
% diff3 = reshape(nanmean(abs(ivrmse3(1:N,:,:)-ivrmse_approx(1:N,:,:)),1),9,9);
% rel_diff3 = reshape(nanmean(abs(ivrmse3(1:N,:,:)-ivrmse_approx(1:N,:,:))./ivrmse3(1:N,:,:),1),9,9);
% nancounter3 = reshape(sum(isnan(ivrmse3),1),9,9);
% nanmean3 = reshape(mean(isnan(ivrmse3),1),9,9);
% %figure('Name','Vega vs differce 2')
% mape_pw3 = abs(ivrmse3(1:N,:,:)-ivrmse_approx(1:N,:,:))./ivrmse3(1:N,:,:);
% %scatter(reshape(vega,81*N,1),reshape(mape_pw3,81*N,1))
% %set(gca, 'XScale', 'log'),set(gca, 'YScale', 'log')
% %
% %%
% N = length(param);
% imp_vola_price_change = zeros(N,9,9);
% imp_vola_price_change2 = zeros(N,9,9);
% for i =1:N
%     if mod(i,100)==0,disp(i);end
%     for m =1:9
%         for s=1:9
%             imp_vola_price_change(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,(0.04*rand(1,1)+0.98)*price(i,m,s));
%             imp_vola_price_change2(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,(0.02*rand(1,1)+0.99)*price(i,m,s));
%         end
%     end
% end
% nan_change = reshape(sum(isnan(imp_vola_price_change),1),9,9);
% nan_change2 = reshape(sum(isnan(imp_vola_price_change2),1),9,9);
% %%
% N = length(param);
% imp_vola_price_change3 = zeros(N,9,9);
% imp_vola_price_change4 = zeros(N,9,9);
% imp_vola_price_change5 = zeros(N,9,9);
% for i =1:N
%     if mod(i,500)==0,disp(i);end
%     for m =1:9
%         for s=1:9
%             imp_vola_price_change3(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,(0.06*rand(1,1)+0.97)*price(i,m,s));
%             imp_vola_price_change4(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,(0.08*rand(1,1)+0.96)*price(i,m,s));
%             imp_vola_price_change5(i,m,s) = blsimpv(S0, Strikes(s),rates(m), Maturities(m)/252,(0.1*rand(1,1)+0.95)*price(i,m,s));
%         end
%     end
% end
% nan_change3 = reshape(sum(isnan(imp_vola_price_change3),1),9,9);
% nan_change4 = reshape(sum(isnan(imp_vola_price_change4),1),9,9);
% nan_change5 = reshape(sum(isnan(imp_vola_price_change5),1),9,9);
% %%
% intrinsic = zeros(N,9,9); 
% intrinsicdiff = zeros(N,9,9); 
% intrinsic_price = zeros(N,9,9);
% for i =1:N
%     for m =1:9
%         for s=1:9
%             intrinsic(i,m,s) = forecast(i,m,s)>max([S0-Strikes(s),0]);
%             intrinsic_price(i,m,s) = price(i,m,s)>max([S0-Strikes(s),0]);
%             intrinsicdiff(i,m,s) = forecast(i,m,s)-S0+Strikes(s);
%         end
%     end
% end
% bigger1 = intrinsicdiff>1;
% int_mean = reshape(mean(intrinsic,1),9,9);
% int_mean2 = 1-int_mean;
% scatter(1:81*N,reshape(intrinsic,1,[])+0.5*reshape(isnan(imp_vola_forecast),1,[]))
% scatter(reshape(intrinsic,1,[]),reshape(imp_vola_forecast,1,[]))
% scatter(reshape(intrinsic,1,[]),reshape(isnan(imp_vola_forecast),1,[]))