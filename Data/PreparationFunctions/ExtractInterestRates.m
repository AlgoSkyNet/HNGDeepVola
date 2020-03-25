clear;
%use the import tool first for the csv file
%create USTreasuryData2001to032020 variable
data = USTreasuryData2001to032020;
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

save('interestRates2001to032020.mat', 'interestRates');