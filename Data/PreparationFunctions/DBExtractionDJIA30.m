clear
%load('../Datasets/SP500/SP500_date_prices_returns.mat')
load('../Datasets/DJIA30/DJIA30_date_prices_returns_220320.mat')

%conn = database('omeos','root','root@123','Vendor','MySQL',...
%                'Server','127.0.0.1', 'PortNumber', 3306);
            
conn = database('MySQL','root','root@123');
            
savefile = 'Puts2014.mat';
                        
query = 'SELECT * FROM omeos WHERE secid = 102456 AND date >= DATE("2014-01-01") AND date <= DATE("2014-12-31") AND (cp_flag ="P") AND exercise_style = "E" AND volume > 100 AND open_interest > 100 AND vega != "NaN" AND div_convention = "I"';     % LIMIT 20';     
%102480 NASDAQ100
%108105 SP500
%102456 DJIA30
% OR cp_flag ="P"  OR cp_flag ="P" 
curs = exec(conn,query);
res = fetch(curs);
data = res.Data;
close(conn);

% 
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
%     index_today=(DJIA30_date_prices_returns(1,:)>=today);
%     index_expiry=(DJIA30_date_prices_returns(1,:)<=expiry);
%     ind_today_exact=find(index_today,1,'first');
%     
%     TradingDaysToMaturity(i)=sum((index_today&index_expiry));
%     Moneyness(i)=DJIA30_date_prices_returns(2,ind_today_exact)/(StrikePriceoftheOptionTimes1000(i)/10);
%     
% end
% TradingDaysToMaturity=TradingDaysToMaturity';
% Moneyness=Moneyness';
% 
% Volume = cell2mat(data(:, 10));
% OpenInterestfortheOption = cell2mat(data(:, 12)); 
% 
% for i=1:NumOptions
%     ind=find(DJIA30_date_prices_returns(1,:)==TheDateofthisPriceInSerialNumber(i));
%     TheDJIA30PriceThisDate(i)=DJIA30_date_prices_returns(2,ind);
%     TheDJIA30ReturnThisDate(i)=DJIA30_date_prices_returns(3,ind);
%     
%     
% end
% TheDJIA30ReturnThisDate = TheDJIA30ReturnThisDate';
% TheDJIA30PriceThisDate = TheDJIA30PriceThisDate';
% HighestClosingBidAcrossAllExchanges = cell2mat(data(:, 9));
% LowestClosingAskAcrossAllExchanges = cell2mat(data(:, 11));
% MeanOptionPrice = [HighestClosingBidAcrossAllExchanges LowestClosingAskAcrossAllExchanges];
% MeanOptionPrice = mean(MeanOptionPrice,2);
% VegaKappaoftheOption = cell2mat(data(:, 17));
% ImpliedVolatilityoftheOption = cell2mat(data(:, 13));
% 
% save(savefile, 'TheDateofthisPrice', 'TheDateofthisPriceInSerialNumber', 'CCallPPut', 'ExpirationDateoftheOption', 'ExpirationDateoftheOptionInSerialNumber',...
%     'TradingDaysToMaturity', 'Moneyness', 'Volume', 'OpenInterestfortheOption', 'StrikePriceoftheOptionTimes1000', ...
%     'TheDJIA30ReturnThisDate', 'TheDJIA30PriceThisDate', 'MeanOptionPrice', 'VegaKappaoftheOption', 'ImpliedVolatilityoftheOption')

% new structure of database as of 2016

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
    index_today=(DJIA30_date_prices_returns(1,:)>=today);
    index_expiry=(DJIA30_date_prices_returns(1,:)<=expiry);
    ind_today_exact=find(index_today,1,'first');
    
    TradingDaysToMaturity(i)=sum((index_today&index_expiry));
    Moneyness(i)=DJIA30_date_prices_returns(2,ind_today_exact)/(StrikePriceoftheOptionTimes1000(i)/10);
    
end
TradingDaysToMaturity=TradingDaysToMaturity';
Moneyness=Moneyness';

Volume = cell2mat(data(:, 11));
OpenInterestfortheOption = cell2mat(data(:, 12)); 

for i=1:NumOptions
    ind=find(DJIA30_date_prices_returns(1,:)==TheDateofthisPriceInSerialNumber(i));
    TheDJIA30PriceThisDate(i)=DJIA30_date_prices_returns(2,ind);
    TheDJIA30ReturnThisDate(i)=DJIA30_date_prices_returns(3,ind);
    
    
end
TheDJIA30ReturnThisDate = TheDJIA30ReturnThisDate';
TheDJIA30PriceThisDate = TheDJIA30PriceThisDate';
HighestClosingBidAcrossAllExchanges = cell2mat(data(:, 9));
LowestClosingAskAcrossAllExchanges = cell2mat(data(:, 10));
MeanOptionPrice = [HighestClosingBidAcrossAllExchanges LowestClosingAskAcrossAllExchanges];
MeanOptionPrice = mean(MeanOptionPrice,2);
VegaKappaoftheOption = cell2mat(data(:, 16));
ImpliedVolatilityoftheOption = cell2mat(data(:, 13));
 
save(savefile, 'TheDateofthisPrice', 'TheDateofthisPriceInSerialNumber', 'CCallPPut', 'ExpirationDateoftheOption', 'ExpirationDateoftheOptionInSerialNumber',...
     'TradingDaysToMaturity', 'Moneyness', 'Volume', 'OpenInterestfortheOption', 'StrikePriceoftheOptionTimes1000', ...
     'TheDJIA30ReturnThisDate', 'TheDJIA30PriceThisDate', 'MeanOptionPrice', 'VegaKappaoftheOption', 'ImpliedVolatilityoftheOption')

