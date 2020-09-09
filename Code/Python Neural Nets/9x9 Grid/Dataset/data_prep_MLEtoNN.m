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


%% Concentate underlying Data
alldata = {};
k = 0;
for y = years
    for goal = goals
        k = k+1;
        file       = strcat(path_data,'params_options_',num2str(y),'_h0_calibrated_',goal,'_interiorPoint_noYield.mat');
        tmp        = load(file);
        alldata{k} = tmp.values;
        year_total(k) =y;
    end
end
%
Ninputs = 0;
for j = 1:k
    for m = 1:length(alldata{1,j})
        if isempty(alldata{1,j}{1,m})
            continue
        end
        Ninputs = Ninputs+1;
        week_vec(Ninputs) = m;
        year_vec(Ninputs) =year_total(j);
        mse(Ninputs,:)    = alldata{1,j}{1,m}.MSE;
        mape(Ninputs,:)   = alldata{1,j}{1,m}.MAPE;
        params(Ninputs,:) = alldata{1,j}{1,m}.hngparams;
        sig2_0(Ninputs)   = alldata{1,j}{1,m}.sig20; 
        yields(Ninputs,:) = alldata{1,j}{1,m}.yields;
        flag(Ninputs)     = alldata{1,j}{1,m}.optispecs.flag;
    end
end
%sig2_0 = sig2_0';
yields_ = yields(:,[1,3:5]);
for i = 1:length(week_vec)
    w   = params(i,1);
    a   = params(i,2);
    b   = params(i,3);
    g   = params(i,4);
    sig = params(i,5);    
    daylengths    = [21,42, 13*5, 126, 252]./252;
    interestRates = yields(i,:);
    notNaN        = ~isnan(interestRates);             
    yieldcurve    = interp1(daylengths(notNaN),interestRates(notNaN),data_vec(:,2)/252);
    yieldcurve(isnan(yieldcurve)) = 0;
    r_cur         = interp1(daylengths(notNaN),interestRates(notNaN),Maturity/252);
    r_cur(isnan(r_cur)) = 0;
    price         = price_Q_clear([w,a,b,g],data_vec,yieldcurve/252,sig);
    yield_matrix(:,i)  = yieldcurve;
    scenario_data(i,:) = [a, b, g, w,sig,r_cur,price];
    constraint(i)      = b+a*g^2;
end
fprintf('%s','Generating Prices completed.'),fprintf('\n')
data_price = scenario_data;
data_9x9 = permute(reshape(data_price(:,15:end),[],9,9),[2,3,1]);

%% Volatility Calculation
price_vec  = zeros(1,Nmaturities*Nstrikes);
bad_idx    = [];
for i = 1:size(data_price,1)
    price_vec = data_price(i,4+1+Nmaturities+1:end);
    vola(i,:) = blsimpv(data_vec(:, 3), data_vec(:, 1), yield_matrix(:,i), data_vec(:, 2)/252,price_vec')';
    if any(isnan(vola(i,:)))
        bad_idx(end+1) = i;
    else
        vega(i,:) = blsvega(data_vec(:,3),  data_vec(:, 1),yield_matrix(:,i), data_vec(:,2)/252, vola(i,:)');
    end
end
fprintf('%s','Generating Volas completed.'),fprintf('\n')
idx               = setxor(1:size(data_price,1),bad_idx);
data_vola         = data_price(:,1:4+1+Nmaturities);
data_vola(:,4+1+Nmaturities+1:95) = vola;
data_vola         = data_vola(idx,:);
data_vega         = vega(idx,:);
data_price        = data_price(idx,:);
constraint        = constraint(idx);
%save('MLE_calib_price.mat','data_price')
%save('MLE_calib_vola.mat','data_vola')
%save('MLE_calib_vega.mat','data_vega')
load("data_fullnormal_intrinsic.mat")


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
comp =[];
comparison ={};
for j=2:51
    price_int = 0;
    idx_cal = j;
    data_week_tmp = data(:,(weeksprices == idx_cal))';
    NN1_surface = reshape(price(idx_cal-1,:,:),9,9)/2000;
    real_data = data_week_tmp(:,[1,2,3,4]);
    for i = 1:length(real_data)
       [pos1,lam1] = conv_comb_maturity(real_data(i,2));
       [pos2,lam2] = conv_comb_strike(real_data(i,3)/real_data(i,4));
       price_surface = real_data(i,4)*NN1_surface;
       %2D interpolation without gridded data
       if pos1~=9 && pos2~=9
            price_int(i) = lam1*lam2*price_surface(pos1,pos2)...
           +(1-lam1)*lam2*price_surface(pos1+1,pos2)...
           +lam1*(1-lam2)*price_surface(pos1,pos2+1)...
           +(1-lam1)*(1-lam2)*price_surface(pos1+1,pos2+1);
       elseif pos1==9 && pos2~=9
              price_int(i) = lam2*price_surface(pos1,pos2)...
           +(1-lam2)*price_surface(pos1,pos2+1);
       elseif pos2==9 && pos1~=9
              price_int(i) = lam1*price_surface(pos1,pos2)...
           +(1-lam1)*price_surface(pos1+1,pos2);
       elseif pos1==9 && pos2==9
              price_int(i) = price_surface(pos1,pos2);
       end
    end
    interestRates = alldata{1, 1}{1, j}.yields; 
    notNaN = ~isnan(interestRates);
    daylengths = [21, 42, 13*5, 126, 252]./252;
    r_cur = interp1(daylengths(notNaN), interestRates(notNaN), data_week_tmp(:, 2)./252);
    r_cur(isnan(r_cur))=0;  
    imp_vola_hng = alldata{1, 1}{1, j}.blsimpvhng;
    imp_vola_real = blsimpv(data_week_tmp(:,4),data_week_tmp(:,3),r_cur, data_week_tmp(:, 2)./252,data_week_tmp(:,1));
    imp_vola_NN = blsimpv(data_week_tmp(:,4),data_week_tmp(:,3),r_cur, data_week_tmp(:, 2)./252,price_int');
    rel_diff_vola1 = abs(imp_vola_NN-imp_vola_real)./imp_vola_real;
    rel_diff_vola2 = abs(imp_vola_hng-imp_vola_real)./imp_vola_real;
    rel_diff_vola3 = abs(imp_vola_NN-imp_vola_hng)./imp_vola_hng;
    diff1 = price_int'-real_data(:,1);
    diff2 = alldata{1, 1}{1, j}.hngPrice'-real_data(:,1);
    rel_diff1 = abs(price_int'-real_data(:,1))./real_data(:,1);
    rel_diff2 = abs(alldata{1, 1}{1, j}.hngPrice'-real_data(:,1))./real_data(:,1);
    rel_diff3 = abs(price_int'-alldata{1, 1}{1, j}.hngPrice')./alldata{1, 1}{1, j}.hngPrice';
    %comp(end+1,:) = [mean(rel_diff),mape(idx_cal-1)];
    comparison{j} = [real_data(:,1),price_int',alldata{1, 1}{1, j}.hngPrice',imp_vola_real,imp_vola_NN,imp_vola_hng,rel_diff1,rel_diff2,rel_diff3,rel_diff_vola1,rel_diff_vola2,rel_diff_vola3,diff1,diff2];
    
end
comp_total = zeros(1,14);
comp_mean = zeros(50,14);
for j=2:51
    tmp = comparison{1,j};
    comp_total(end+1:end+length(tmp),:) = tmp;
    comp_mean(j,:) = nanmean(tmp,1);
end
comp_total = comp_total(2:end,:);
comp_mean = comp_mean(2:end,7:12);
subplot(2,1,1)
plot(1:50,comp_mean(:,[1,2]));
legend("MAPE NNvsObs","MAPE HNGvsObs");%"MAPE NNvsHNG")
title("Mean Price Deviation for every week in 2010")
%set(gca, 'YScale', 'log')
subplot(2,1,2)
plot(1:50,comp_mean(:,[4,5]));
legend("IV-MAPE NNvsObs","IV-MAPE HNGvsObs")%,"IV-MAPE NNvsHNG")
title("Mean IV Deviation for every week in 2010")

%set(gca, 'YScale', 'log')

