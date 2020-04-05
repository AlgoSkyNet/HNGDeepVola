% Dataset Generator
% This Program generates the dataset for our neural net.
% Each file has a 12 digit unique id for identification 4 files are saved: 
% 1 mat-file with parameters and prices. 
% 1 mat-file with parameters and imp volas.
% 1 mat-file which summarizes the characteristics and specifics of the generator
% 1 png-file with histograms. 
% The filenames are always of the following structure:
% id _ {12digit id} _ {type of file}
% The two dataset files have the parameter type and size added to the name:
% id _ {12digit id} _ {type of file} _ {type of params} _ {Numbers Szenarios}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%           THIS FILE NEEDS MATLAB VERSION 2018a OR HIGHER TO RUN         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars; clc;close all;
id =  java.util.UUID.randomUUID;id = char(id.toString);id=convertCharsToStrings(id([1:8,10:13,15:18]));

%% Initialisation

% Configuration of underlying data
years     = 2010:2017;
goals     = ["MSE"];%,"MAPE","OptLL"];
path_data = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/Calibration Calloption/';

% Configuration of dataset
%rng('default') % in case we want to check results set to fixed state
choice          = "norm"; % 1."norm" 2."uni" 3."unisemiscale" 4."log" 5."tanh" 6."tanhscale"
yieldstype      = "szenario"; % "PCA" only! "szenario" not working yet.
Maturity        = 30:30:210;
K               = 0.9:0.025:1.1;
S               = 1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';
Nsim            = 100000;

% At the moment, to ensure good pseudo random numbers, all randoms numbers are drawn at once.
% Hence it is only possible to specify the total number of draws (Nsim). 
% The approx. size of the final dataset is 14% of Nsim for norm dist and
% 10% for uni dist

%% TODO (sorted by importance)
% pca yieldcurves only positive!
% Smarter way for incorperation of yieldcurve to the neural net
% Nsim fix


%% Concentate underlying Data
alldata = {};
k = 0;
for y = years
    for goal = goals
        k = k+1;
        file       = strcat(path_data,'params_options_',num2str(y),'_h0_calibrated_',goal,'_InteriorPoint_noYield.mat');
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


%% Dataset Generator
normalizer = @(input) (input-repmat(mean(input),length(input),1))./repmat(std(input),length(input),1);
inv_scaler = @(input,my,sig) input.*repmat(sig,length(input),1)+repmat(my,length(input),1);
%data_pure  = [params,sig2_0,yields_];
data_pure  = [params,yields_];

% Choosing good parameters
if strcmp(choice,"norm") || strcmp(choice,"uni") || strcmp(choice,"unisemiscale")
    data  = data_pure;
elseif strcmp(choice,"log")
    data  = [log(data_pure(:,1:3)),data_pure(:,4),log(data_pure(:,5)),data_pure(:,6:end)];
elseif strcmp(choice,"tanh") || strcmp(choice,"tanhscale")
    data2 = (2*(data_pure-(10e-8)*sign(data_pure-repmat(mean(data_pure),length(data_pure),1)))-repmat([max(data_pure)+min(data_pure)],length(data_pure),1))./(repmat([max(data_pure)-min(data_pure)],length(data_pure),1));
    data  = atanh(data2);
end

mean_     = mean(data);
std_      = std(data);
data_norm = normalizer(data);
[coeff,score,~,~,explained,mu] =pca(data_norm(:,6:9));
expl_sum = cumsum(explained);
check = -1;
for i = 1:length(coeff)
    if check==1
        continue
    else
        if expl_sum(i)>95
            ind   = i;
            check = 1;
        end
    end
end
trafo_yields     = score(:,1:ind);
trafo_yields_std = std(trafo_yields);
trafo_data       = [data_norm(:,1:5),trafo_yields./trafo_yields_std];       
cov_trafo        = cov(trafo_data);
mean_trafo       = mean(trafo_data);
sample           = mvnrnd(mean_trafo,cov_trafo,Nsim);
yields_inv       = repmat(mu,length(sample),1)+trafo_yields_std.*sample(:,6:end)*coeff(:,1:ind)';
sample_trafo     = [sample(:,1:5),yields_inv];

if strcmp(choice,"norm") 
    inv_data   = inv_scaler(sample_trafo,mean(data_pure),std(data_pure));
elseif strcmp(choice,"uni")
    inv_data   = inv_scaler(sample_trafo,mean(data_pure),std(data_pure));
    yields_tmp = inv_data(:,6:end);
    %inv_data   = [min([params,sig2_0])+(max([params,sig2_0])-min([params,sig2_0])).*rand(Nsim,5),yields_tmp];
    inv_data   = [min([params])+(max([params])-min([params])).*rand(Nsim,5),yields_tmp];
elseif strcmp(choice,"unisemiscale")
    inv_data   = inv_scaler(sample_trafo,mean(data_pure),std(data_pure));
    yields_tmp = inv_data(:,6:end);
    %inv_data   = [min([params,sig2_0])+(max([params,sig2_0])-min([params,sig2_0])).*rand(Nsim,5),yields_tmp];
    inv_data   = [min([params])+(max([params])-min([params])).*rand(Nsim,5),yields_tmp];
    inv_data   = inv_data./std(inv_data).*std(data_pure);
elseif strcmp(choice,"log")
    inv_data   = [exp(sample_trafo(:,1:3)),sample_trafo(:,4),exp(sample_trafo(:,5)),yields_inv];
    inv_data   = inv_scaler(normalize(inv_data),mean(data_pure),std(data_pure));
elseif strcmp(choice,"tanh")
    inv_data   = inv_scaler(sample_trafo,mean_,std_);
    inv_data   = tanh(inv_data);
    inv_data   = 0.5*(inv_data.*(repmat([max(data_pure)-min(data_pure)],length(inv_data),1))+repmat([max(data_pure)+min(data_pure)],length(inv_data),1));
elseif strcmp(choice,"tanhscale")
    inv_data   = inv_scaler(sample_trafo,mean_,std_);
    inv_data   = inv_scaler(normalize(tanh(inv_data)),mean(data2),std(data2));
    inv_data   = 0.5*(inv_data.*(repmat([max(data_pure)-min(data_pure)],length(inv_data),1))+repmat([max(data_pure)+min(data_pure)],length(inv_data),1));
end

if strcmp(yieldstype,"szenario")
    i_rand = randi(Ninputs,Nsim,1);
end
inv_data = [inv_data(:,1:5),max(inv_data(:,6:end),0)];

% Price Calculations
j = 0;
fail1 = 0;
fail2 = 0;
fail3 = 0;
yield_matrix  = zeros(Nmaturities*Nstrikes,Nsim);
scenario_data = zeros(Nsim,Nmaturities*Nstrikes+5+Nmaturities);
constraint    = zeros(1,Nsim); 
if strcmp(yieldstype,"PCA")
    yields_clean  = zeros(Nsim,4);
end
fprintf('%s','Generating Prices. Progress: 0%')
for i = 1:Nsim
    if ismember(i,floor(Nsim*[4/100:4/100:1]))
        fprintf('%0.5g',round(i/(Nsim)*100,0)),fprintf('%s',"%")
    end
    w   = inv_data(i,1);
    a   = inv_data(i,2);
    b   = inv_data(i,3);
    g   = inv_data(i,4);
    sig = inv_data(i,5);
    if w<=0 || a<0 || b<0 || sig<=0
        fail3 = fail3+1;
        continue
    end
    if b+a*g^2 >= 1 
        fail1 = fail1+1;
        continue
    end
    if strcmp(yieldstype,"szenario")
        int = i_rand(i);
        daylengths    = [21,42, 13*5, 126, 252]./252;
        interestRates = yields(int,:);
        notNaN        = ~isnan(interestRates);             
        yieldcurve    = interp1(daylengths(notNaN),interestRates(notNaN),data_vec(:,2)/252);
        r_cur         = interp1(daylengths(notNaN),interestRates(notNaN),Maturity/252);
        price         = price_Q_clear([w,a,b,g],data_vec,yieldcurve/252,sig);
    elseif strcmp(yieldstype,"PCA")
        daylengths    = [21,13*5, 126, 252]./252;
        interestRates = inv_data(i,6:end);
        yieldcurve    = interp1(daylengths,interestRates,data_vec(:,2)/252);
        r_cur         = interp1(daylengths,interestRates,Maturity/252);
        price         = price_Q_clear([w,a,b,g],data_vec,yieldcurve/252,sig);
    end
    if any(any(price <= 0)) || any(any(isnan(price)))
        fail2 = fail2+1;
        continue
    end
    j=j+1;
    if strcmp(yieldstype,"PCA")
        yields_clean(j,:) = inv_data(i,6:end); 
    end
    yield_matrix(:,j)  = yieldcurve;
    scenario_data(j,:) = [a, b, g, w,sig,r_cur,price];
    constraint(j)      = b+a*g^2;
end
yield_matrix  = yield_matrix(:,1:j);
scenario_data = scenario_data(1:j,:);
constraint    = constraint(1:j); 
fprintf('%s','Generating Prices completed.'),fprintf('\n')
data_price = scenario_data;
if strcmp(yieldstype,"PCA")
    yields_clean = yields_clean(1:j,:);
end

%% Volatility Calculation
price_vec  = zeros(1,Nmaturities*Nstrikes);
bad_idx    = [];
fprintf('%s','Calculating Imp Volas. Progress: 0%')
for i = 1:size(data_price,1)
     if ismember(i,floor(j*[10/100:10/100:1]))
        fprintf('%0.5g',round(i/(j)*100,0)),fprintf('%s',"%")
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
if strcmp(yieldstype,"PCA")
    yields_clean = yields_clean(idx,:);
end
save(strcat('id_',id,'_data_price_',choice,'_',num2str(size(data_price,1)),'.mat'),'data_price')
save(strcat('id_',id,'_data_vola_',choice,'_',num2str(size(data_vola,1)),'.mat'),'data_vola')


%% Summary and Visualisation for control purposes

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
figure("Name",id)
subplot(2,3,1),histogram(data_vola(:,1),'Normalization','probability');title('alpha')
subplot(2,3,2),histogram(data_vola(:,2),'Normalization','probability');title('beta')
subplot(2,3,3),histogram(data_vola(:,3),'Normalization','probability');title('gamma')
subplot(2,3,4),histogram(data_vola(:,4),'Normalization','probability');title('omega')
subplot(2,3,5),histogram(data_vola(:,5),'Normalization','probability');title('sigma')
subplot(2,3,6),histogram(constraint,'Normalization','probability');title('constraint');
saveas(gcf,strcat('id_',id,'_histograms','.png'))

% Example plot
%figure
%[X,Y]=meshgrid(K,Maturity);
%surf(X',Y',reshape(data_price(1,4+1+Nmaturities+1:end),9,7));hold on;
%scatter3(data_vec(:,1),data_vec(:,2),scenario_data(1,4+1+Nmaturities+1:end));