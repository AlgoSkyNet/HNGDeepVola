%%
clear;
%use the import tool first for the csv file
%create USTreasuryYieldData2001to032020 variable
data = USTreasuryYieldData2001to032020;
% names of columns 
% date	1 mo	3 mo	6 mo	1 yr	2 yr	3 yr	5 yr	7 yr	10 yr

dates = datenum(table2array(data(:,1)));
% 1 month
monthly(:,1) = (table2array(data(:,2)))/100;
% 3 months
monthly(:,2) = (table2array(data(:,3)))/100;
% 6 months
monthly(:,3) = (table2array(data(:,4)))/100;
% 12 months
monthly(:,4) = (table2array(data(:,5)))/100;

% interestRates
interestRates(1, :) = dates';
interestRates(2:5, :) = monthly';

save('interestRatesYield2001to032020.mat', 'interestRates');

%%
clear;
%use the import tool first for the csv file
%create USTreasuryTbillData2001to032020 variable
data = USTreasuryTbillData2001to032020;
% names of columns 
% 1 - date	
% 2 - 4 weeks bank discount	
% 3 - 4 weeks coupon equivalent
% 4 - 8 weeks bank discount	
% 5 - 8 weeks coupon equivalent
% 6 - 13 weeks bank discount	
% 7 - 13 weeks coupon equivalent
% 8 - 26 weeks bank discount	
% 9 - 26 weeks coupon equivalent
% 10 - 52 weeks bank discount	
% 11 - 52 weeks coupon equivalent
temp = USTreasuryTbillData2001to1.VarName1;
d = datetime(temp,'InputFormat','MM/dd/yy','Format','dd-MMM-yyyy');
dates = datenum(datestr(d));
% 4 weeks bank discount	
monthly(:,1) = (table2array(data(:,2)))/100;
% 8 weeks bank discount	
temp = table2cell((data(:,4)));
array = [];
for i = 1:length(temp)
    if temp{i}=='N/A'
        array(i, 1) = NaN;
    else
        array(i, 1) = double(string(temp{i}));
    end
end
monthly(:,2) = array/100;
% 13 weeks bank discount	
monthly(:,3) = (table2array(data(:,6)))/100;
% 26 weeks bank discount
monthly(:,4) = (table2array(data(:,8)))/100;
% 52 weeks bank discount
temp = table2cell((data(:,10)));
array = [];
for i = 1:length(temp)
    if temp{i}=='N/A'
        array(i, 1) = NaN;
    else
        array(i, 1) = double(string(temp{i}));
    end
end
monthly(:,5) = array/100;

% interestRates
interestRates(1, :) = dates';
interestRates(2:6, :) = monthly';

save('interestRatesTbill2001to032020.mat', 'interestRates');