% Dataset Generator
% This Program generates the dataset for our neural net.

clearvars; clc;close all;
%temp =  java.util.UUID.randomUUID;myuuid = temp.toString;disp(myuuid)81a21cb5-105e-40d1-9823-69d75586e389

%% Initialisation
%concentate all datasets.
path_data = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/final option calibration/';
alldata   = {};
k = 0;
for y=2010:2011%2018
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
choice          = "uni";
%% Dataset Generation 
Nsim            = 2000;

% Choosing good parameters

if strcmp(choice,"norm")
    % Scaled Normal distribution
    rand_params = mvnrnd(mean_,cov_,Nsim);
elseif strcmp(choice,"uni")
    % uniform distributio
    rand_params = min_+(max_-min_).*rand(Nsim,5);
end
% Choosing a termstructure out of the giving structures
i_rand = randi(Ninputs,Nsim,1);


% Price Calculations
j = 0;
fail1 =0;
fail2 =0;
fail3 =0;
fprintf('%s','Generating Prices. Progress: 0%')
for i = 1:Nsim
    w   = rand_params(i,1);
    a   = rand_params(i,2);
    b   = rand_params(i,3);
    g   = rand_params(i,4);
    sig = rand_params(i,5);
    int = i_rand(i);
    if w<=0 || a<0 || b<0 || sig<=0
        fail3 = fail3+1;
        continue
    end
    if b+a*g^2 >= 1 
        fail1 = fail1+1;
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
    if ismember(i,floor(Nsim*[4/100:4/100:1]))
        fprintf('%0.5g',round(i/(Nsim)*100,1)),fprintf('%s',"%")
    end
    yield_matrix(:,j) = yieldcurve;
    scenario_data(j,:) = [a, b, g, w,sig,r_cur,price];
    constraint(j) = b+a*g^2;
end
fprintf('%s','Generating Prices completed.'),fprintf('\n')

data_price = scenario_data;
%%
% Volatility Calculation
price_vec  = zeros(1,Nmaturities*Nstrikes);
bad_idx    = [];
fprintf('%s','Calculating Imp Volas. Progress: 0%')
for i = 1:size(data_price,1)
     if ismember(i,floor(j*[10/100:10/100:1]))
        fprintf('%0.5g',round(i/(j)*100,1)),fprintf('%s',"%")
     end
    price_vec = data_price(i,4+1+Nmaturities+1:end);
    vola(i,:) = blsimpv(data_vec(:, 3),  data_vec(:, 1), yield_matrix(:,i), data_vec(:, 2)/252,price_vec')';
    if any(isnan(vola(i,:))) || any(vola(i,:)==0)
        bad_idx(end+1) = i;
    end
end
fprintf('%s','Generating Volas completed.'),fprintf('\n')
idx               = setxor(1:size(data_price,1),bad_idx);
data_vola         = data_price(:,1:4+1+Nmaturities);
data_vola(:,4+1+Nmaturities+1:75) = vola;
data_vola         = data_vola(idx,:);
data_price        = data_price(idx,:);
constraint        = constraint(idx);
save(strcat('data_price_',choice,'_',num2str(size(data_price,1)),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity)),'.mat'),'data_price')
save(strcat('data_vola_',choice,'_',num2str(size(data_vola,1)),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity)),'.mat'),'data_vola')

idx = logical(idx);

%% Visualisation of control purposes
% Summary
prices = data_price(:,4+1+Nmaturities+1:end);
volas  = data_vola(:,4+1+Nmaturities+1:end);
param  = data_vola(:,1:5);
tab_data = [Nsim,length(idx),fail1/Nsim,fail3/Nsim,fail2/Nsim,length(bad_idx)/Nsim,...
    max(prices,[],'all'),min(prices,[],'all'),mean(prices,'all'),median(prices,'all'),...
    max(volas,[],'all'),min(volas,[],'all'),mean(volas,'all'),median(volas,'all'),...
    median(param),median(constraint)];
stat = array2table(tab_data);    
stat.Properties.VariableNames = {'Nsim','Nfinal','fail_con','fail_pos','fail_prices','fail_volas','max_price','min_price','mean_price','median_price',...
    'max_vola','min_vola','mean_vola','median_vola','median_alpha','median_beta','median_gamma','median_omega','median_sigma2_0','median_con'};
stat = rows2vars(stat);
stat.Properties.VariableNames = {'Property','Value'}; 
disp(stat)
save(strcat('summary_',choice,num2str(size(data_vola,1)),'.mat'),'stat')   
figure
subplot(2,3,1),histogram(scenario_data(idx,1),'Normalization','probability');title('alpha')
subplot(2,3,2),histogram(scenario_data(idx,2),'Normalization','probability');title('beta')
subplot(2,3,3),histogram(scenario_data(idx,3),'Normalization','probability');title('gamma')
subplot(2,3,4),histogram(scenario_data(idx,4),'Normalization','probability');title('omega')
subplot(2,3,5),histogram(scenario_data(idx,5),'Normalization','probability');title('sigma')
subplot(2,3,6),histogram(constraint(idx),'Normalization','probability');title('constraint');
saveas(gcf,strcat('histograms_',choice,num2str(size(data_vola,1)),'.png'))

% Example plot
%figure
%[X,Y]=meshgrid(K,Maturity);
%surf(X',Y',reshape(data_price(1,4+1+Nmaturities+1:end),9,7));hold on;
%scatter3(data_vec(:,1),data_vec(:,2),scenario_data(1,4+1+Nmaturities+1:end));