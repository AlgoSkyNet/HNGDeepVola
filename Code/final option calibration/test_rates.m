tbill = load("C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets/InterestRates/SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat");
tbill = tbill.SP500_date_prices_returns_realizedvariance_interestRates;  
yield  =load("C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets/InterestRates/SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat");
yield = yield.SP500_date_prices_returns_realizedvariance_interestRates;  
column = any(tbill<=0);
logi = tbill<=0;
nanvec = any(isnan(tbill));
row = any(tbill<=0,2);
feasible = and(~nanvec,column);
data = tbill(:,feasible);
plot(tbill(:,feasible))
column2 = any(tbill(5:end,:)<=0);
column3 = any(tbill(5:end,2:end)<=0);
column4 = any(tbill(5:end,1:end-1)<=0);
sum(and(and(column2(2:end-1),column3(1:end-1)),column4(2:end)))