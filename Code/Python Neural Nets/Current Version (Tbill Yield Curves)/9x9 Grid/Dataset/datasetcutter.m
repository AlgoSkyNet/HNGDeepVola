% Dataset Cutter
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
threshold = 10e-8;
idx = ~any(data_price(:,15:end)'<threshold);
data_price = data_price(idx,:);
data_vola = data_vola(idx,:);
save("id_dfc18d626cbb42f1_data_vola_norm_cutted.mat","data_vola")
save("id_dfc18d626cbb42f1_data_price_norm_cutted.mat","data_price")
