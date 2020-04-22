%% Collection of different tools to modify a dataset
clearvars,clc
load("id_dfc18d626cbb42f1_data_vola_norm_205851clean.mat")
load("id_dfc18d626cbb42f1_data_price_norm_205851clean.mat")
load("id_dfc18d626cbb42f1_summary.mat")

Maturity        = 10:30:250;%30:30:210  10:30:250
K               = 0.9:0.025:1.1;
S               = 1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';
N               = length(data_price);

%% small price cutter
threshold = 10e-8;
idx = ~any(data_price(:,15:end)'<threshold);
data_price = data_price(idx,:);
data_vola = data_vola(idx,:);
save("id_dfc18d626cbb42f1_data_vola_norm_cutted.mat","data_vola")
save("id_dfc18d626cbb42f1_data_price_norm_cutted.mat","data_price")
%% vega calculation

data_vega = zeros(length(data_price),Nmaturities*Nstrikes);
for i =1:length(data_price)
    yield_curve = repmat(data_vola(i,6:14),1,Nstrikes);
    data_vega(i,:) =  blsvega(data_vec(:,3),  data_vec(:, 1),yield_curve', data_vec(:,2)/252, data_vola(i,4+1+Nmaturities+1:95)');
end
data_vega_cut = data_vega(idx,:);
save("id_dfc18d626cbb42f1_vega.mat","data_vega")
save("id_dfc18d626cbb42f1_vega_cutted.mat","data_vega_cut")
