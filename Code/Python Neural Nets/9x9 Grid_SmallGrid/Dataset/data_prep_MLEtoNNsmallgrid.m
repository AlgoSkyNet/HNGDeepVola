clearvars,clc;
years     = 2010:2018;
goals     = ["MSE"];%,"MAPE","OptLL"];
%path_data = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Calibration Calloption/';
%path_data = 'D:/GitHub/HNGDeepVola/Code/Calibration Calloption/';
path_data = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Calibration Calloption/';

%C:\Users\Henrik\Documents\GitHub\HNGDeepVola\Code\calibration checks\Results for the note\Tables\calibration under Q\data for tables\results calibr h0Calibrated esth0P\MSE

Maturity        = 10:30:250;%30:30:210  10:30:250
Moneyness               = 1.1:-0.025:0.9;
S               = 2000;%1;
K               = S./Moneyness;
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
%% price calculation for every week
for i = 1:length(week_vec)
    if mod(i,100)==0
        disp(i);
    end
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
%save('MLE_calib_price_full.mat','data_price')
%save('MLE_calib_vola_full.mat','data_vola')
%save('MLE_calib_vega_full.mat','data_vega')

%% load NN forecasts
load("data_intrinsic_smallgird.mat")

%% get call data

%Options: path = '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
warning('on')

%parpool()
path                = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Data/Datasets';
%path                = 'D:/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
%path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
%path                =  'C:/Users/TEMP/Documents/GIT/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';

useYield            = 0; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 0; % use last h_t from MLE under P as h0
num_voladays        = 6; % if real vola, give the number of historic volas used (6 corresponds to today plus 5 days = 1week);
algorithm           = 'interior-point';% 'sqp'
data =[];
weekyear =[];
for year = 2010:2018
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
    disp(length(unique(weeksprices)))
    year_idx = year*ones(1,length(weeksprices));
    idxj  = 1:length(unique(weeksprices));
    weakyear_tmp = [weeksprices;year_idx];
    weekyear = [weekyear,weakyear_tmp];
    data_year = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega; OptionsStruct.implied_volatility];
    data = [data,data_year];
end
%% Compare performance
comp =[];
short_table ={};
comparison ={};
iv_table ={};
run_idx = 0;
run_idx2 = 0;
for year = 2010:2018
    for j=1:53
        data_week_tmp = data(:,and((weekyear(1,:) == j),(weekyear(2,:) == year)))';
        if isempty(data_week_tmp)
           %disp([j,year])
           continue
        else
            %disp([j,year]);
            run_idx = run_idx+1;      
            run_idx2 = run_idx2+1;
            if ismember(run_idx,bad_idx)
                run_idx2 = run_idx2-1;
                continue
            else
                NN1_surface = reshape(price(run_idx2,:,:),9,9)/2000;
                real_data = data_week_tmp(:,[1,2,3,4]);
                price_int = zeros(1,length(real_data));
                for i = 1:length(real_data)
                   [pos1,lam1] = conv_comb_maturity_small(real_data(i,2));
                   [pos2,lam2] = conv_comb_strike_moneyness(real_data(i,4)/real_data(i,3));
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
                interestRates = alldata{1, year-2009}{1, j}.yields; 
                notNaN = ~isnan(interestRates);
                daylengths = [21, 42, 13*5, 126, 252]./252;
                r_cur = interp1(daylengths(notNaN), interestRates(notNaN), data_week_tmp(:, 2)./252);
                r_cur(isnan(r_cur))=0;  
                imp_vola_hng = alldata{1, year-2009}{1, j}.blsimpvhng;
                imp_vola_real = blsimpv(data_week_tmp(:,4),data_week_tmp(:,3),r_cur, data_week_tmp(:, 2)./252,data_week_tmp(:,1));
                imp_vola_NN = blsimpv(data_week_tmp(:,4),data_week_tmp(:,3),r_cur, data_week_tmp(:, 2)./252,price_int');
                rel_diff_vola1 = abs(imp_vola_NN-imp_vola_real)./imp_vola_real;
                rel_diff_vola2 = abs(imp_vola_hng-imp_vola_real)./imp_vola_real;
                rel_diff_vola3 = abs(imp_vola_NN-imp_vola_hng)./imp_vola_hng;
                diff1 = price_int'-real_data(:,1);
                diff2 = alldata{1, year-2009}{1, j}.hngPrice'-real_data(:,1);
                rel_diff1 = abs(price_int'-real_data(:,1))./real_data(:,1);
                rel_diff2 = abs(alldata{1, year-2009}{1, j}.hngPrice'-real_data(:,1))./real_data(:,1);
                rel_diff3 = abs(price_int'-alldata{1, year-2009}{1, j}.hngPrice')./alldata{1, year-2009}{1, j}.hngPrice';
                intrinsic_value = data_week_tmp(:,4)-data_week_tmp(:,3).*exp(-r_cur.*data_week_tmp(:,2)/252);
                intrinsic_value(intrinsic_value<0)=0;
                intrinsic_sign =  real_data(:,1)>=intrinsic_value;
                ivrmse1 = (imp_vola_NN-imp_vola_real).^2;
                ivrmse2 = (imp_vola_hng-imp_vola_real).^2;
                ivrmse3 = (imp_vola_NN-imp_vola_hng).^2;
                ivapp1 = ((price_int'-real_data(:,1))./data_week_tmp(:,5)).^2;
                ivapp2 = ((alldata{1, year-2009}{1, j}.hngPrice'-real_data(:,1))./data_week_tmp(:,5)).^2;
                vega_hng =  blsvega(data_week_tmp(:,4),  data_week_tmp(:,3),r_cur,data_week_tmp(:,2)/252, imp_vola_hng);
                ivapp3 = ((price_int'-alldata{1, year-2009}{1, j}.hngPrice')./vega_hng).^2;
                rel_diff_iv1 = abs((ivapp1 -ivrmse1)./ivrmse1);
                rel_diff_iv2 = abs((ivapp2 -ivrmse2)./ivrmse2);
                rel_diff_iv3 = abs((ivapp3 -ivrmse3)./ivrmse3);
                iv_table{run_idx2} =[ivrmse1,ivrmse2,ivrmse3,ivapp1,ivapp2,ivapp3,rel_diff_iv1,rel_diff_iv2,rel_diff_iv3];
                %comp(end+1,:) = [mean(rel_diff),mape(idx_cal-1)];
                comparison{run_idx2} = [real_data(:,1),price_int',alldata{1, year-2009}{1, j}.hngPrice',imp_vola_real,imp_vola_NN,imp_vola_hng,rel_diff1,rel_diff2,rel_diff3,...
                    rel_diff_vola1,rel_diff_vola2,rel_diff_vola3,diff1,diff2,...
                    real_data(:,2),real_data(:,3)./real_data(:,4),intrinsic_value,intrinsic_sign,];
                short_table{run_idx2} = [data_week_tmp(:,1:4),intrinsic_value,intrinsic_sign,r_cur];
            end
        end
    end
end

comp_total = zeros(1,18);
comp_mean = zeros(50,18);
total_short = zeros(1,7);
iv_weakly = zeros(50,9);
for j=1:run_idx2
    tmp = comparison{1,j};
    tmp_idx = tmp(:,15)>0;
    tmp = tmp(tmp_idx,:);
    tmp2 = short_table{1,j};
    tmp2 = tmp2(tmp_idx,:);
    tmpiv = iv_table{1,j};
    tmpiv = tmpiv(tmp_idx,:);
    iv_weakly(j,:) = nanmean(tmpiv);
    iv_weakly(j,1:6) = sqrt(iv_weakly(j,1:6));
    comp_total(end+1:end+size(tmp,1),:) = tmp;
    comp_mean(j,:) = nanmean(tmp,1);
    total_short(end+1:end+size(tmp,1),:) = tmp2;
end

total_short = total_short(2:end,:);
comp_total = comp_total(2:end,:);
comp_mean = comp_mean(1:end,7:12);
%%
figure
subplot(2,1,1)
plot(1:run_idx2,comp_mean(:,[1,2]));
legend("MAPE NNvsObs","MAPE HNGvsObs");%"MAPE NNvsHNG")
title("Mean Price Deviation for every week in 2010")
%set(gca, 'YScale', 'log')
subplot(2,1,2)
plot(1:run_idx2,comp_mean(:,[4,5]));
legend("IV-MAPE NNvsObs","IV-MAPE HNGvsObs")%,"IV-MAPE NNvsHNG")
title("Mean IV Deviation for every week in 2010")
%set(gca, 'YScale', 'log')
figure
subplot(3,1,1)
scatter(comp_total(:,1),comp_total(:,7),"x");hold on
scatter(comp_total(:,1),comp_total(:,8),"x");
set(gca, 'XScale', 'log')
legend("MAPE NNvsObs","MAPE HNGvsObs");%"MAPE NNvsHNG")
xlabel("log Price")
subplot(3,1,2)
scatter(total_short(:,2),comp_total(:,7),"x");hold on
scatter(total_short(:,2),comp_total(:,8),"x");
legend("MAPE NNvsObs","MAPE HNGvsObs");%"MAPE NNvsHNG")
xlabel("DtM")
subplot(3,1,3)
scatter(total_short(:,4)./total_short(:,3),comp_total(:,7),"x");hold on
scatter(total_short(:,4)./total_short(:,3),comp_total(:,8),"x");
legend("MAPE NNvsObs","MAPE HNGvsObs");%"MAPE NNvsHNG")
xlabel("Moneyness")


bad_Values = total_short((comp_total(:,7)-comp_total(:,8))>0.4,:);
