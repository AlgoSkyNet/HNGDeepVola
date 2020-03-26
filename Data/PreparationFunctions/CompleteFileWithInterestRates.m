%%
clear;
load('SP500_date_prices_returns_realizedvariance_090320.mat');
data = SP500_date_prices_returns_realizedvariance_090320;

load('interestRatesYield2001to032020.mat');
data(5:8,:) = NaN;
[~, T] = size(interestRates);
lastDate = data(1,end);
for i = 1:T
    if interestRates(1, i) <= lastDate
        if i == 29
            aa=0;
        end
        ind = find(data(1,:) == interestRates(1, i));
        
        if ~isempty(ind)
            data(5:8, ind) = interestRates(2:end,i);
            indtemp = ind;
        else
            data(5:8, indtemp + 1) = data(5:8, indtemp);
            indtemp = indtemp + 1;
        end
    end
end
SP500_date_prices_returns_realizedvariance_interestRates = data;
save('SP500_date_prices_returns_realizedvariance_interestRatesYield_090320.mat', 'SP500_date_prices_returns_realizedvariance_interestRates');

%%
clear;
load('SP500_date_prices_returns_realizedvariance_090320.mat');
data = SP500_date_prices_returns_realizedvariance_090320;

load('interestRatesTbill2001to032020.mat');
data(5:9,:) = NaN;
[~, T] = size(interestRates);
lastDate = data(1,end);
for i = 1:T
    if interestRates(1, i) <= lastDate
        if i == 29
            aa=0;
        end
        ind = find(data(1,:) == interestRates(1, i));
        
        if ~isempty(ind)
            data(5:9, ind) = interestRates(2:end,i);
            indtemp = ind;
        else
            data(5:9, indtemp + 1) = data(5:9, indtemp);
            indtemp = indtemp + 1;
        end
    end
end
SP500_date_prices_returns_realizedvariance_interestRates = data;
save('SP500_date_prices_returns_realizedvariance_interestRatesTbill_090320.mat', 'SP500_date_prices_returns_realizedvariance_interestRates');