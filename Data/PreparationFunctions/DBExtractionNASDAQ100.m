clear
%load('../Datasets/SP500/SP500_date_prices_returns.mat')
load('../Datasets/NASDAQ100/NASDAQ100_date_prices_returns_220320.mat')


% conn = database('sandbox','root','root','Vendor','MySQL',...
%                 'Server','127.0.01', 'PortNumber', 3307);

conn = database('MySQL','root','root@123');

savefile = 'CallsPuts2015.mat';
            
query = 'SELECT * FROM omeos WHERE secid = 102480 AND date >= DATE("2015-01-01") AND date <= DATE("2015-12-31") AND (cp_flag ="C" or cp_flag ="P") AND exercise_style = "E" AND volume > 100 AND open_interest > 100 AND vega != "NaN" AND div_convention = "I"';     % LIMIT 20';     
%108105 SP500
%102456 DJIA
%102480 NASDAQ100
curs = exec(conn,query);
res = fetch(curs);
data = res.Data;
close(conn);
% TheDateofthisPrice = data(:, 2);
% TheDateofthisPriceInSerialNumber = datenum(TheDateofthisPrice, 'yyyy-mm-dd'); 
% CCallPPut = data(:, 6);
% ExpirationDateoftheOption = data(:, 5);
% ExpirationDateoftheOptionInSerialNumber = datenum(ExpirationDateoftheOption, 'yyyy-mm-dd');
% % Compute time to maturities
% NumOptions=length(ExpirationDateoftheOptionInSerialNumber);
% StrikePriceoftheOptionTimes1000 = cell2mat(data(:, 8));
% for i=1:NumOptions
%     today=TheDateofthisPriceInSerialNumber(i);
%     expiry=ExpirationDateoftheOptionInSerialNumber(i);
%     index_today=(NASDAQ100_date_prices_returns(1,:)>=today);
%     index_expiry=(NASDAQ100_date_prices_returns(1,:)<=expiry);
%     ind_today_exact=find(index_today,1,'first');
%     
%     TradingDaysToMaturity(i)=sum((index_today&index_expiry));
%     Moneyness(i)=NASDAQ100_date_prices_returns(2,ind_today_exact)/(StrikePriceoftheOptionTimes1000(i)/1000);
%     
% end
% TradingDaysToMaturity=TradingDaysToMaturity';
% Moneyness=Moneyness';
% 
% Volume = cell2mat(data(:, 10));
% OpenInterestfortheOption = cell2mat(data(:, 12)); 
% 
% for i=1:NumOptions
%     ind=find(NASDAQ100_date_prices_returns(1,:)==TheDateofthisPriceInSerialNumber(i));
%     TheNASDAQ100PriceThisDate(i)=NASDAQ100_date_prices_returns(2,ind);
%     TheNASDAQ100ReturnThisDate(i)=NASDAQ100_date_prices_returns(3,ind);
%     
%     
% end
% TheNASDAQ100ReturnThisDate = TheNASDAQ100ReturnThisDate';
% TheNASDAQ100PriceThisDate = TheNASDAQ100PriceThisDate';
% HighestClosingBidAcrossAllExchanges = cell2mat(data(:, 9));
% LowestClosingAskAcrossAllExchanges = cell2mat(data(:, 11));
% MeanOptionPrice = [HighestClosingBidAcrossAllExchanges LowestClosingAskAcrossAllExchanges];
% MeanOptionPrice = mean(MeanOptionPrice,2);
% VegaKappaoftheOption = cell2mat(data(:, 17));
% ImpliedVolatilityoftheOption = cell2mat(data(:, 13));
% 
% save(savefile, 'TheDateofthisPrice', 'TheDateofthisPriceInSerialNumber', 'CCallPPut', 'ExpirationDateoftheOption', 'ExpirationDateoftheOptionInSerialNumber',...
%     'TradingDaysToMaturity', 'Moneyness', 'Volume', 'OpenInterestfortheOption', 'StrikePriceoftheOptionTimes1000', ...
%     'TheNASDAQ100ReturnThisDate', 'TheNASDAQ100PriceThisDate', 'MeanOptionPrice', 'VegaKappaoftheOption', 'ImpliedVolatilityoftheOption')


TheDateofthisPrice = data(:, 2);
TheDateofthisPriceInSerialNumber = datenum(TheDateofthisPrice, 'yyyy-mm-dd'); 
CCallPPut = data(:, 7);
ExpirationDateoftheOption = data(:, 5);
ExpirationDateoftheOptionInSerialNumber = datenum(ExpirationDateoftheOption, 'yyyy-mm-dd');
% Compute time to maturities
NumOptions=length(ExpirationDateoftheOptionInSerialNumber);
StrikePriceoftheOptionTimes1000 = cell2mat(data(:, 8));
for i=1:NumOptions
    today=TheDateofthisPriceInSerialNumber(i);
    expiry=ExpirationDateoftheOptionInSerialNumber(i);
    index_today=(NASDAQ100_date_prices_returns(1,:)>=today);
    index_expiry=(NASDAQ100_date_prices_returns(1,:)<=expiry);
    ind_today_exact=find(index_today,1,'first');
    
    TradingDaysToMaturity(i)=sum((index_today&index_expiry));
    Moneyness(i)=NASDAQ100_date_prices_returns(2,ind_today_exact)/(StrikePriceoftheOptionTimes1000(i)/1000);
    
end
TradingDaysToMaturity=TradingDaysToMaturity';
Moneyness=Moneyness';

Volume = cell2mat(data(:, 11));
OpenInterestfortheOption = cell2mat(data(:, 12)); 

for i=1:NumOptions
    ind=find(NASDAQ100_date_prices_returns(1,:)==TheDateofthisPriceInSerialNumber(i));
    TheNASDAQ100PriceThisDate(i)=NASDAQ100_date_prices_returns(2,ind);
    TheNASDAQ100ReturnThisDate(i)=NASDAQ100_date_prices_returns(3,ind);
    
    
end
TheNASDAQ100ReturnThisDate = TheNASDAQ100ReturnThisDate';
TheNASDAQ100PriceThisDate = TheNASDAQ100PriceThisDate';
HighestClosingBidAcrossAllExchanges = cell2mat(data(:, 9));
LowestClosingAskAcrossAllExchanges = cell2mat(data(:, 10));
MeanOptionPrice = [HighestClosingBidAcrossAllExchanges LowestClosingAskAcrossAllExchanges];
MeanOptionPrice = mean(MeanOptionPrice,2);
VegaKappaoftheOption = cell2mat(data(:, 16));
ImpliedVolatilityoftheOption = cell2mat(data(:, 13));

save(savefile, 'TheDateofthisPrice', 'TheDateofthisPriceInSerialNumber', 'CCallPPut', 'ExpirationDateoftheOption', 'ExpirationDateoftheOptionInSerialNumber',...
    'TradingDaysToMaturity', 'Moneyness', 'Volume', 'OpenInterestfortheOption', 'StrikePriceoftheOptionTimes1000', ...
    'TheNASDAQ100ReturnThisDate', 'TheNASDAQ100PriceThisDate', 'MeanOptionPrice', 'VegaKappaoftheOption', 'ImpliedVolatilityoftheOption')