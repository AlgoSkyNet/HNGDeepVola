% Dataset Generator
% This Program generates the dataset for our neural net.

clearvars; clc;close all;

%% Initialisation
%concentate all datasets.
path_data = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/final option calibration/';
alldata   = {};
k = 0;
for y=2010%:2018
    for goal = ["MSE","MAPE","OptLL"]
        k = k+1;
        file    = strcat(path_data,'params_Options_',num2str(y),'_h0asRealVola_',goal,'_InteriorPoint_noYield.mat');
        tmp     = load(file);
        alldata{k} = tmp.values;
    end
end
%
Ninputs = 0;
for j = 1:k
    for m=1:length(alldata{1,j})
        if isempty(alldata{1,j}{1,m})
            continue
        end
        Ninputs = Ninputs+1;
        params(Ninputs,:) = alldata{1,j}{1,m}.hngparams;
        sig2_0(Ninputs)   = alldata{1,j}{1,m}.sig20; 
        yields(Ninputs,:) = alldata{1,j}{1,m}.yields;
    end
end
sig2_0 = sig2_0';

% normalised data 
cov_        = cov([params,sig2_0]);
mean_       = mean([params,sig2_0]);
%init_norm   = normalize([params,sig2_0]);
%cov_norm    = cov(norm_matrix);
min_        = min([params,sig2_0]);
max_        = max([params,sig2_0]);

%rng('default')
Maturity        = 30:30:210;
K               = 0.9:0.025:1.1;
S               = 1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';

%% Dataset Generation 
Nsim            = 120000;

% Choosing good parameters
% Scaled Normal distribution
rand_params = mvnrnd(mean_,cov_,Nsim);
% uniform distributio
% rand_parans = min_+(max_-min_)*rand(Nsim,5)

% Choosing a termstructure out of the giving structures
i_rand = randi(Ninputs,Nsim,1);


% Price Calculations
j = 0;
fail1 =0;
fail2 =0;
fail3 =0;
fprintf('%s','Generatiting Prices. Progress: 0%')
for i = 1:Nsim
    w   = rand_params(i,1);
    a   = rand_params(i,2);
    b   = rand_params(i,3);
    g   = rand_params(i,1);
    sig = rand_params(i,1);
    int = i_rand(i);
    if b+a*g^2 >= 1 
        fail1 = fail1+1;
        continue
    end
    if w<=0 || a<0 || b<0 || sig<=0
        fail3 = fail3+1;
        continue
    end
    daylengths = [21,42, 13*5, 126, 252]./252;
    interestRates = yields(int,:);
    notNaN = ~isnan(interestRates);             
    yieldcurve = interp1(daylengths(notNaN),interestRates(notNaN),data_vec(:,2)/252);
    r_cur = interp1(daylengths(notNaN),interestRates(notNaN),Maturity/252);
    price   = price_Q_clear([w,a,b,g],data_vec,yieldcurve/252,sig);
    if any(any(price <= 0)) || any(any(isnan(price)))
        fail2 = fail2+1;
        continue
    end
    j=j+1;
    if ismember(i,floor(Nsim*[1/100:1/100:1]))
        fprintf('%0.5g',round(i/(Nsim)*100,1)),fprintf('%s',"%"),fprintf('\n')
    end
    yield_matrix(:,j) = yieldcurve;
    scenario_data(j,:) = [a, b, g, w,sig,r_cur,price];
    constraint(j) = 1-b-a*g^2;
end
data_price = scenario_data;

% Volatility Calculation
price_vec  = zeros(1,Nmaturities*Nstrikes);
bad_idx    = [];
fprintf('%s','Calculating Imp Volas. Progress: 0%')
for i = 1:j
     if ismember(i,floor(j*[1/100:1/100:1]))
        fprintf('%0.5g',round(i/(j)*100,1)),fprintf('%s',"%"),fprintf('\n')
     end
    price_vec = data_price(i,4+1+Nmaturities+1:end);
    vola(i,:) = blsimpv(data_vec(:, 3),  data_vec(:, 1), yield_matrix(:,i), data_vec(:, 2)/252,price_vec')';
    if any(isnan(vola(i,:))) || any(vola(i,:)==0)
        bad_idx(end+1) = i;
    end
end 
idx               = setxor(1:j,bad_idx);
data_vola         = data_price(:,1:4+1+Nmaturities);
data_vola(:,4+1+Nmaturities+1:75) = vola;
data_vola         = data_vola(idx,:);
data_price        = data_price(idx,:);
save(strcat('data_price_','maxbounds','_',num2str(size(data_price,1)),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity)),'.mat'),'data_price')
save(strcat('data_vola_','maxbounds','_',num2str(size(data_vola,1)),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity)),'.mat'),'data_vola')

idx = logical(idx);
fail3 = 0;
%% Visualisation of control purposes
% Summary
    fprintf('\n')
    disp(['Summary:'])
    disp(['failed szenarios (constraint): ', num2str(round(100*fail1/Nsim,2)),'%'])
    disp(['failed szenarios (positivity): ', num2str(round(100*fail3/Nsim,2)),'%'])
    disp(['failed szenarios (bad prices):   ', num2str(round(100*fail2/Nsim,2)),'%'])
    disp(['failed szenarios (bad volas):    ', num2str(round(100*length(bad_idx)/Nsim,2)),'%'])
    disp(['max price:     ', num2str(max(data_price(:,4+1+Nmaturities+1:end),[],'all'))])
    disp(['min price:     ', num2str(min(data_price(:,4+1+Nmaturities+1:end),[],'all'))])
    disp(['mean price:    ', num2str(mean(data_price(:,4+1+Nmaturities+1:end),'all'))])
    disp(['median price:  ', num2str(median(data_price(:,4+1+Nmaturities+1:end),'all'))])
    disp(['max vola:      ', num2str(max(data_vola(:,4+1+Nmaturities+1:end),[],'all'))])
    disp(['min vola:      ', num2str(min(data_vola(:,4+1+Nmaturities+1:end),[],'all'))])
    disp(['mean vola:     ', num2str(mean(data_vola(:,4+1+Nmaturities+1:end),'all'))])
    disp(['median vola:   ', num2str(median(data_vola(:,4+1+Nmaturities+1:end),'all'))])
    disp(['median alpha:  ', num2str(median(scenario_data(idx,1)))])
    disp(['median beta:   ', num2str(median(scenario_data(idx,2)))])
    disp(['median gamma:  ', num2str(median(scenario_data(idx,3)))])
    disp(['median omega:  ', num2str(median(scenario_data(idx,4)))])
    disp(['median sigma:  ', num2str(median(scenario_data(idx,5)))])
prices = data_price(:,4+1+Nmaturities+1:end);
volas  = data_vola(:,4+1+Nmaturities+1:end);
param  = data_vola(:,1:5);
tab_data = [Nsim,length(idx),round(100*fail1/Nsim,2),round(100*fail3/Nsim,2),round(100*fail2/Nsim,2),round(100*length(bad_idx)/Nsim,2),...
    max(prices,[],'all'),min(prices,[],'all'),mean(prices,'all'),median(prices,'all'),...
    max(volas,[],'all'),min(volas,[],'all'),mean(volas,'all'),median(volas,'all'),...
    median(param)];
stat = array2table(tab_data);    
stat.Properties.VariableNames = {'Nsim','Nfinal','fail_con','fail_pos','fail_prices','fail_volas','max_price','min_price','mean_price','median_price',...
    'max_vola','min_vola','mean_vola','median_vola','median_alpha','median_beta','median_gamma','median_omega','median_sigma2_0'};
stat = rows2vars(stat);
stat.Properties.VariableNames = {'Property','Value'}; 
save(strcat('summarydataset',num2str(size(data_vola,1)),'.mat'),'stat')   
figure
subplot(2,3,1),histogram(scenario_data(idx,1),'Normalization','probability');title('alpha')
subplot(2,3,2),histogram(scenario_data(idx,2),'Normalization','probability');title('beta')
subplot(2,3,3),histogram(scenario_data(idx,3),'Normalization','probability');title('gamma')
subplot(2,3,4),histogram(scenario_data(idx,4),'Normalization','probability');title('omega')
subplot(2,3,5),histogram(scenario_data(idx,5),'Normalization','probability');title('sigma')
subplot(2,3,6),histogram(constraint(idx),'Normalization','probability');title('constraint');
saveas(gcf,strcat('histogramsdataset',num2str(size(data_vola,1)),'.png'))

% Example plot
%figure
%[X,Y]=meshgrid(K,Maturity);
%surf(X',Y',reshape(data_price(1,4+1+Nmaturities+1:end),9,7));hold on;
%scatter3(data_vec(:,1),data_vec(:,2),scenario_data(1,4+1+Nmaturities+1:end));