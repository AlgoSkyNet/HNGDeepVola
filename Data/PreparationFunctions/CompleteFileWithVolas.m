clear;
load('SPX_volas_090320.mat');
t = datetime(table2array(SPX(:,1)),'InputFormat','yyyy-MM-dd HH:mm:ss+ss:ss');
data_oxford(:,1) = datenum(datestr(datetime(table2array(SPX(:,1)),'InputFormat','yyyy-MM-dd HH:mm:ss+ss:ss')));
data_oxford(:,2) = table2array(SPX(:,11));
data_oxford(:,3) = table2array(SPX(:,12));
data_oxford(:,4) = table2array(SPX(:,21));
data_oxford = data_oxford';

load('SP500_date_prices_returns_090320.mat');
SP500_date_prices_returns_realizedvariance_090320 = SP500_date_prices_returns;
SP500_date_prices_returns_realizedvariance_090320(4,:) = NaN;
[~, T] = size(data_oxford);
for i = 1:T
    ind = find(SP500_date_prices_returns_realizedvariance_090320(1,:)==data_oxford(1, i));
    SP500_date_prices_returns_realizedvariance_090320(4, ind) = data_oxford(3,i);
end
    
save('SP500_date_prices_returns_realizedvariance_090320.mat', 'SP500_date_prices_returns_realizedvariance_090320');