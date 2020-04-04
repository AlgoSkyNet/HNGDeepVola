clear all;
% all puts and calls
%year_files = {'/Users/Lukas/Documents/GitHub/SeminarOptions/Data/Datasets/SP500/Calls2014.mat'};
%year_files  =  {'/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2018.mat'};
year_files = {'/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2010.mat', ...
    '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2011.mat',...
    '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2012.mat', ...
    '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2013.mat',...
    '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2014.mat', ...
    '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2015.mat',...
    '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2016.mat',...
    '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2017.mat',...
    '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets/SP500/Calls2018.mat'};
formatIn = 'dd-mmm-yyyy';


NumYears = length(year_files);




Type = ['call'];
MinimumVolume = 100;
MinimumOpenInterest = 100;
IfCleanNans = 1;

MaturitiesBounds = [7 30 80 180 250];
MoneynessBounds = [.9-1e-9 .95 .975 1 1.025 1.05 1.1];

NumberOfContracts = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);
AveragePrices = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);
AverageImpliedVolatilities = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);

num_year = 2010;

for k = 1:NumYears
    load(year_files{k});
    DateString_start    = sprintf('01-January-%d', num_year);
    DateString_end      = sprintf('31-December-%d', num_year);
    date_start          = datenum(DateString_start,formatIn);
    date_end            = datenum(DateString_end,formatIn);
    Dates               = [date_start:1:date_end];
    % For Wednesdays
    wednesdays          = (weekday(Dates)==4);
    index               = find(wednesdays);
    Dates               = Dates(index)';

    
    for i=1:length(MaturitiesBounds) - 1
        for j=1:length(MoneynessBounds) - 1
            TimeToMaturityInterval = [MaturitiesBounds(i) MaturitiesBounds(i + 1)];
            MoneynessInterval = [MoneynessBounds(j) MoneynessBounds(j + 1)];
            OptionData = SelectOptions(Dates, Type, TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans, TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume,OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);
            
            if length(OptionData)
                NumberOfContracts(i, j) = NumberOfContracts(i, j) + length(OptionData);
                Prices_temp = 0;
                ImpliedVolatilities_temp = 0;
                for r = 1:length(OptionData)
                    Prices_temp = Prices_temp + OptionData(r).price;
                    ImpliedVolatilities_temp = ImpliedVolatilities_temp + OptionData(r).implied_volatility;
                end
                AveragePrices(i, j) = AveragePrices(i, j) + Prices_temp;
                AverageImpliedVolatilities(i, j) = AverageImpliedVolatilities(i, j) + ImpliedVolatilities_temp;
            end
        end
    end  
    num_year = num_year + 1;
end

AveragePrices = AveragePrices ./ NumberOfContracts;
AverageImpliedVolatilities = AverageImpliedVolatilities ./ NumberOfContracts;

% load('params_test_options_2012_h0asRealVola_MSE_interiorpoint_noYield');
% 
% impl=[];
% j = 1;
% for i=1:length(values)
%     if ~isempty(values{1,i})
%         cur_length = length(values{1,i}.blsimpv);
%         impl(j:j + cur_length - 1) = values{1,i}.blsimpv;
%         j = j + cur_length;
%     end
% end

%save('description_data_pricing_thursdays_2015.mat'); % or wednesdays
save('description_data_pricing_calls_wednesdays_2010_2018.mat');