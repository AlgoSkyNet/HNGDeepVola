clearvars,clc;
years     = 2010:2018;
goals     = ["MSE"];%,"MAPE","OptLL"];
%path_data = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Calibration Calloption/';
%path_data = 'D:/GitHub/HNGDeepVola/Code/Calibration Calloption/';
path_data = 'D:/GitHub/MasterThesisHNGDeepVola/Code/Calibration Calloption/';

%C:\Users\Henrik\Documents\GitHub\HNGDeepVola\Code\calibration checks\Results for the note\Tables\calibration under Q\data for tables\results calibr h0Calibrated esth0P\MSE

Maturity        = 10:30:250;%30:30:210  10:30:250
K               = 0.9:0.025:1.1;
S               = 2000;%1;
K               = K*S;
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
%path                = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Data/Datasets';
path                = 'D:/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
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
    surface_for_python =  flipud(reshape(interpolatedData,9,9)');
    % VERSION 2 MAXGRID INTERPOLATION > MISSING VALUES
    max_strike = floor(40*max(strike_norm./S))/40;
    max_mat = floor((max(data_week_tmp(:,2))-10)/30)*30+10;
    min_strike =ceil(40*min(strike_norm./S))/40;
    min_mat = ceil((min(data_week_tmp(:,2))-10)/30)*30+10;
    K2 = S.*[min_strike:0.025:max_strike];
    Mat2 = min_mat:30:max_mat;
    gridpoints2 = combvec(K2,Mat2)';
    interpolatedData2 = interFunc(gridpoints2);
    interpolatedData_surf = reshape(interpolatedData2,length(K2),length(Mat2))';
    surface_for_python2 = -999*ones(9,9);
    pos_x1 =  conv_comb_maturity(min_mat);
    pos_x2 = conv_comb_maturity(max_mat);
    pos_y1 =  conv_comb_strike(min_strike);
    pos_y2 = conv_comb_strike(max_strike);
    surface_for_python2(pos_x1:pos_x2,pos_y1:pos_y2) = interpolatedData_surf;
    surface_for_python2 = flipud(surface_for_python2);
    mv_idx = surface_for_python2==-999;
    surface_forNN = cat(3,surface_for_python2,mv_idx);
    data_1(j-1,:,:) = surface_for_python;
    data_2(j-1,:,:,:) = surface_forNN;
end
save("2010_interpolatedgrid_full.mat","data_1");
save("2010_interpolatedgrid_mv.mat","data_2");