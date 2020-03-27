% Dataset Generator
% This Program generates the dataset for our neural net.
% Each file has a 12 digit unique id for identification
% 4 files are saved: 1 file with parameters and prices. 1 file with
% parameters and imp volas. 1 file which summarizes the characteristics and
% specifics of the generation and 1 png-file with histograms. 
% The filenames are always of the following structure: id _ {12digit id} _ {type of file}
% For the files with data the name ends with the number of szenarios:
% id _ {12digit id} _ {type of file} _ {Numbers Szenarios}

% At the moment, to ensure good pseudo random numbers, all randoms numbers are drawn at once.
% Hence it is only possible to specify the total number of draws (Nsim). The
% approx. size of the final dataset is 17% of Nsim.
clearvars; clc;close all;

%% Initialisation
%concentate all datasets.
path_data = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/final option calibration/';
alldata   = {};
k = 0;
years = 2010:2011;
goals = ["MSE","MAPE","OptLL"];
for y=years
    for goal = goals
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

choice          = "norm"; %"norm"  "uni"

id =  java.util.UUID.randomUUID;id = char(id.toString);id=convertCharsToStrings(id([1:8,10:13,15:18]));

%% Dataset Generation 
Nsim            = 300000;

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
    if ismember(i,floor(Nsim*[4/100:4/100:1]))
        fprintf('%0.5g',round(i/(Nsim)*100,1)),fprintf('%s',"%")
    end
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
save(strcat('id_',id,'_data_price_',num2str(size(data_price,1)),'.mat'),'data_price')
save(strcat('id_',id,'_data_vola_',num2str(size(data_vola,1)),'.mat'),'data_vola')

idx = logical(idx);

%% Summary and Visualisation of control purposes
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
stat = table2struct(stat);
stat.param_type = choice;
stat.years = years;
stat.goalfuncs = goals;
stat.Maturities = Maturity;       
stat.Moneyness  = K;
stat.id =id;
disp(stat)
save(strcat('id_',id,'_summary','.mat'),'stat')   
figure
subplot(2,3,1),histogram(scenario_data(idx,1),'Normalization','probability');title('alpha')
subplot(2,3,2),histogram(scenario_data(idx,2),'Normalization','probability');title('beta')
subplot(2,3,3),histogram(scenario_data(idx,3),'Normalization','probability');title('gamma')
subplot(2,3,4),histogram(scenario_data(idx,4),'Normalization','probability');title('omega')
subplot(2,3,5),histogram(scenario_data(idx,5),'Normalization','probability');title('sigma')
subplot(2,3,6),histogram(constraint(idx),'Normalization','probability');title('constraint');
saveas(gcf,strcat('id_',id,'_histograms','.png'))

% Example plot
%figure
%[X,Y]=meshgrid(K,Maturity);
%surf(X',Y',reshape(data_price(1,4+1+Nmaturities+1:end),9,7));hold on;
%scatter3(data_vec(:,1),data_vec(:,2),scenario_data(1,4+1+Nmaturities+1:end));