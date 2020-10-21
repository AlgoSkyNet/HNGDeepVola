clearvars,clc;
years     = 2010:2018;
goals     = ["MSE"];%,"MAPE","OptLL"];
path_data = 'C:/Users/Henrik/Documents/GitHub/HNGDeepVola/Code/Calibration Calloption/';
%path_data = 'D:/GitHub/HNGDeepVola/Code/Calibration Calloption/';
%path_data = 'D:/GitHub/MasterThesisHNGDeepVola/Code/Calibration Calloption/';

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
j = 0;
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
    if ~any(price<1e-6)
        j = j+1;
        yield_matrix(:,j)  = yieldcurve;
        scenario_data(j,:) = [a, b, g, w,sig,r_cur,price];
        constraint(j)      = b+a*g^2;
    end
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
save('MLE_calib_price_Moneyness.mat','data_price')
save('MLE_calib_vola_Moneyness.mat','data_vola')
save('MLE_calib_vega_Moneyness.mat','data_vega')
