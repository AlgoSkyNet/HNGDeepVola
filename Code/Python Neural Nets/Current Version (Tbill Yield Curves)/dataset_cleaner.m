%dataset_cleaner
clc,clearvars
load ("id_3283354135d44b67_data_vola_norm_231046clean.mat")
load ("id_3283354135d44b67_data_price_norm_231046clean.mat")
j=1;
for i =1:length(data_vola)
    if mod(i,100)==0
        disp(i/length(data_vola))
    end
    if any(data_vola(i,13:end)>1)|| any(data_price(i,13:end)>1)
        continue
    else
        data_vola_new(j,:) = data_vola(i,:);
        data_price_new(j,:) = data_price(i,:);
        j = j+1;
    end
end