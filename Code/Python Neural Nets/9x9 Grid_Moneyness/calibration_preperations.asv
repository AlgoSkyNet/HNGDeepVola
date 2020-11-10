clearvars,clc;
years     = 2010:2018;
goals     = ["MSE"];%,"MAPE","OptLL"];
path_data = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Calibration Calloption/';
%path_data = 'D:/GitHub/HNGDeepVola/Code/Calibration Calloption/';
%path_data = 'D:/GitHub/MasterThesisHNGDeepVola_Moneyness/Code/Calibration Calloption/';

%C:\Users\Henrik\Documents\GitHub\HNGDeepVola\Code\calibration checks\Results for the note\Tables\calibration under Q\data for tables\results calibr h0Calibrated esth0P\MSE

Maturity        = 10:30:250;%30:30:210  10:30:250
Moneyness       = 1.1:-0.025:0.9;
S               = 2000;%1;
K               = S./Moneyness;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';
% At the moment, to ensure good pseudo random numbers, all randoms numbers are drawn at once.
% Hence it is only possible to specify the total number of draws (Nsim). 
% The approx. size of the final dataset is 14% of Nsim for norm dist and
% 10% for uni dist


%Options: path = '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
warning('on')

%parpool()
path                = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Data/Datasets';
%path                = 'D:/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
%path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
%path                =  'C:/Users/TEMP/Documents/GIT/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';
year                = 2010;
useYield            = 0; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 0; % use last h_t from MLE under P as h0
num_voladays        = 6; % if real vola, give the number of historic volas used (6 corresponds to today plus 5 days = 1week);
algorithm           = 'interior-point';% 'sqp'
goal                =  'OptLL'; % 'MSE';   'MAPE';  ,'OptLL';
path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);

% load Interest rates
% load the corresponding data
if useYield
    path_vola       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateYield_090320.mat');
    txt = 'useYield';
else
    path_vola       =  strcat(path, '/', 'InterestRates', '/', 'SP500_date_prices_returns_realizedvariance_intRateTbill_090320.mat');
    txt = 'noYield';
end
load(path_vola);

% if use realized volatility data then load the corresponding data


bound                   = [100, 100];
formatIn                = 'dd-mmm-yyyy';

% start from the first Wednesday of 2015 and finish with the last Wednesday of 2015

DateString_start        = strcat('01-January-',num2str(year));
DateString_end          = strcat('31-December-',num2str(year));
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
wednessdays             = (weekday(date_start:date_end)==4);
Dates                   = date_start:date_end;
Dates                   = Dates(wednessdays);


    
% bounds for maturity, moneyness, volumes, interest rates
Type                    = 'call';
MinimumVolume           = 100;
MinimumOpenInterest     = 100;
IfCleanNans             = 1;
TimeToMaturityInterval  = [8, 250];
MoneynessInterval       = [0.9, 1.1];
[OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptions(Dates, Type, ...
    TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans,...
    TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume, ...
    OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
    TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);

weeksprices             = week(datetime([OptionsStruct.date], 'ConvertFrom', 'datenum'));

idxj  = 1:length(unique(weeksprices));

data = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega; OptionsStruct.implied_volatility];
data_week_1 = data(:,(weeksprices == 1))';
data_week_2 = data(:,(weeksprices == 2))';
%%
load("Dataset/Moneyness_price_norm_400_1e-6.mat")

%%
for j=2:51  
    idx_cal = j;
    data_week_tmp = data(:,(weeksprices == idx_cal))';
    price_week_norm = S.*data_week_tmp(:,1)./data_week_tmp(:,4);
    strike_norm  =  S.*data_week_tmp(:,3)./data_week_tmp(:,4);
    
    interFunc = scatteredInterpolant(strike_norm,data_week_tmp(:,2),price_week_norm);
    real_data_tmp = [strike_norm,data_week_tmp(:,2),price_week_norm];
    % VERSION 1 FULLGRID INTERPOLATION
    gridpoints = combvec(K,Maturity)';
    interpolatedData = interFunc(gridpoints);
    idx = (gridpoints(:,1)<=max(strike_norm)).*(gridpoints(:,1)>=min(strike_norm)).*(gridpoints(:,2)<=max(data_week_tmp(:,2))).*(gridpoints(:,2)>=min(data_week_tmp(:,2)));
    idx = logical(idx);
    interpolatedData2 = interpolatedData(idx);
    surface_for_python2 = -999*ones(81,1);
    surface_for_python2(idx) = interpolatedData(idx);
    surface_for_python2 = reshape( surface_for_python2,9,9)';
    surface_for_python =  reshape(interpolatedData,9,9)';
    %figure
    %heatmap(surface_for_python);
    %heatmap(surface_for_python2);
    % Finding closest matrix
    for i =1:length(data_price)
        price_dataset = data_price(i,15:end)';
        l2norm(i) = mean((price_dataset(idx)-interpolatedData2).^2);
        mape(i) =  100*mean(abs((price_dataset(idx)-interpolatedData2)./interpolatedData2));
    end
    [~,min_idx]= min(l2norm);
    [~,min_idx2]= min(mape);
    %figure 
    %heatmap(reshape(data_price(min_idx,15:end),9,9)')
    %figure 
    %heatmap(reshape(data_price(min_idx2,15:end),9,9)')
    filled_data = data_price(min_idx2,15:end);
    filled_data(idx) = interpolatedData2;
    vola_data = blsimpv(data_vec(:, 3), data_vec(:, 1),reshape(repmat(data_price(min_idx2,6:14),9,1),81,1), data_vec(:, 2)/252,filled_data')';
    rates(j-1,:) = data_price(min_idx2,6:14);
    data_vola(j-1,:,:) = reshape(vola_data,9,9)';
    data_3(j-1,:,:) = reshape(filled_data,9,9)';
    %figure 
    %heatmap(reshape(filled_dat,9,9)');    
    mv_idx = surface_for_python2==-999;
    surface_forNN = cat(3,surface_for_python2,mv_idx);
    data_1(j-1,:,:) = surface_for_python;
    data_2(j-1,:,:,:) = surface_forNN;
end
%save("2010_interpolatedgrid_full.mat","data_1");
%save("2010_interpolatedgrid_mv.mat","data_2");
save("2010_interpolatedgrid_filledvalues.mat","data_3","rates","data_vola");