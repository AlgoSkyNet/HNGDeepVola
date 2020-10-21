clc; clearvars;
load("data_for_IV_moneyness.mat")
load("data_for_IVvola_moneyness.mat")
N = length(param);
S0 = 2000;
Moneyness = 1.1:-0.025:0.9;
Strikes = S0./Moneyness;
Maturities = [10:30:250];
imp_price = zeros(N,9,9);
imp_price_forecast = zeros(N,9,9);
imp_vola = zeros(N,9,9);
imp_vola_forecast = zeros(N,9,9);
for i =1:N
    if mod(i,100)==0,disp(i);end
    for m =1:9
        for s=1:9
            imp_price(i,m,s) = blsprice(S0, Strikes(s),rates(i,m), Maturities(m)/252,vola(i,m,s));
            imp_price_forecast(i,m,s) = blsprice(S0, Strikes(s),rates(i,m), Maturities(m)/252,vola_forecast(i,m,s));
            imp_vola(i,m,s) = blsimpv(S0, Strikes(s),rates(i,m), Maturities(m)/252,price(i,m,s));
            imp_vola_forecast(i,m,s) = blsimpv(S0, Strikes(s),rates(i,m), Maturities(m)/252,forecast(i,m,s));
         end
    end
end
%
mean_p = reshape(mean(abs(price-forecast)./price,1),9,9);
mean_v = reshape(mean(abs(vola-vola_forecast)./vola,1),9,9); 

% Errors from vola network >implied prices
mape1_impprice = abs(price-imp_price_forecast)./price;
mape2_impprice = abs(imp_price-imp_price_forecast)./imp_price;
mean_mape1_impprice = reshape(mean(mape1_impprice,1),9,9);
mean_mape2_impprice = reshape(mean(mape2_impprice,1),9,9);
mse1_impprice = (price-imp_price_forecast).^2;
mse2_impprice = (imp_price-imp_price_forecast).^2;
mean_mse1_impprice = reshape(mean(mse1_impprice,1),9,9);
mean_mse2_impprice = reshape(mean(mse2_impprice,1),9,9);
% Errors from price network >implied volas
mape1= abs(vola-imp_vola_forecast)./vola;
mape2 = abs(imp_vola-imp_vola_forecast)./imp_vola;
mean_mape1 = reshape(mean(mape1,1),9,9);
mean_mape2 = reshape(mean(mape2,1),9,9);
mse1 = (vola-imp_vola_forecast).^2;
mse2 = (imp_vola-imp_vola_forecast).^2;
mean_mse1 = reshape(mean(mse1,1),9,9);
mean_mse2 = reshape(mean(mse2,1),9,9);
