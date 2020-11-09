% Programm to check whether our data is correct and recreatable
clc; clearvars;
load("id_aa11a111a1aa1a1a_data_vola_norm_143032.mat")
load("id_aa11a111a1aa1a1a_data_vega_norm_143032.mat")
load("id_aa11a111a1aa1a1a_data_price_norm_143032.mat")

N = length(data_price);
S0 = 1;
Strikes = S0*[0.9:0.025:1.1];
Maturities = [10:30:250];
params = data_price(:,1:5);
rates = data_price(:,6:14);
prices = data_price(:,15:end);
volas = data_vola(:,15:end);
data_vec = [combvec(Strikes,Maturities);S0*ones(1,81)]';
rates_vec =zeros(N,81);
for i = 1:9
    rates_vec(:,(i-1)*9+1:9*i) = repmat(rates(:,i),1,9);
end
price_new = zeros(N,81);
for i =1:1000
    if mod(i,100)==0,disp(i);    end
    price_new(i,:) = price_Q_clear([params(i,4),params(i,1:3)],data_vec, rates_vec(i,:)'./252,params(i,5));
end
mape = abs(price_new-prices)./prices;
mean_mape =reshape(mean(mape(1:1000,:),1),9,9);

vola_new = zeros(N,81);
for i =1:1000
    if mod(i,100)==0,disp(i);    end
     vola_new(i,:) = blsimpv(data_vec(:, 3), data_vec(:, 1), rates_vec(i,:)', data_vec(:, 2)./252,prices(i,:)');
end
mape_vola = abs(vola_new-volas)./volas;
mean_mape_vola =reshape(mean(mape_vola(1:1000,:),1),9,9);
