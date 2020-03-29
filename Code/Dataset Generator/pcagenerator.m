clearvars; clc;close all;
% Initialisation

% Configuration of underlying data
years     = 2010:2018;
goals     = ["MSE","MAPE","OptLL"];
path_data = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/Calibration Calloption/';

% Configuration of dataset
%rng('default') % in case we want to check results set to fixed state
Nsim            = 1000000;
alldata = {};
k = 0;
for y = years
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
    for m = 1:length(alldata{1,j})
        if isempty(alldata{1,j}{1,m})
            continue
        end
        Ninputs = Ninputs+1;
        mse(Ninputs,:)    = alldata{1,j}{1,m}.MSE;
        mape(Ninputs,:)   = alldata{1,j}{1,m}.MAPE;
        params(Ninputs,:) = alldata{1,j}{1,m}.hngparams;
        sig2_0(Ninputs)   = alldata{1,j}{1,m}.sig20; 
        yields(Ninputs,:) = alldata{1,j}{1,m}.yields;
    end
end
sig2_0 = sig2_0';
yields = yields(:,[1,3:5]);

%% normalised data
normalizer = @(input) (input-repmat(mean(input),length(input),1))./repmat(std(input),length(input),1);
inv_scaler = @(input,my,sig) input.*repmat(sig,length(input),1)+repmat(my,length(input),1);

data_pure = [params,sig2_0,yields];
data = (2*(data_pure-(10e-8)*sign(data_pure-repmat(mean(data_pure),length(data_pure),1)))-repmat([max(data_pure)+min(data_pure)],length(data_pure),1))./(repmat([max(data_pure)-min(data_pure)],length(data_pure),1));
data_log = atanh(data);
%data_log = [log(data_pure(:,1:3)),data_pure(:,4),log(data_pure(:,5)),data_pure(:,6:end)];
mean_ = mean(data_log);
std_ = std(data_log);
data_norm =normalizer(data_log);
[coeff,score,~,~,explained,mu] =pca(data_norm(:,6:9));
expl_sum = cumsum(explained);
check = -1;
for i = 1:length(coeff)
    if check==1
        continue
    else
        if expl_sum(i)>95
            ind = i;
            check = 1;
        end
    end
end
trafo_yields = score(:,1:ind);
trafo_yields_std = std(trafo_yields);
trafo_data = [data_norm(:,1:5),trafo_yields./trafo_yields_std];       
cov_trafo = cov(trafo_data);
mean_trafo = mean(trafo_data);
sample = mvnrnd(mean_trafo,cov_trafo,Nsim);
yields_inv = repmat(mu,length(sample),1)+trafo_yields_std.*sample(:,6:end)*coeff(:,1:ind)';
sample_trafo = [sample(:,1:5),yields_inv];
inv_data = inv_scaler(sample_trafo,mean_,std_);
%inv_data = [exp(inv_data(:,1:3)),inv_data(:,4),exp(inv_data(:,5)),yields_inv];
inv_data = tanh(inv_data);
inv_data = 0.5*(inv_data.*(repmat([max(data_pure)-min(data_pure)],length(inv_data),1))+repmat([max(data_pure)+min(data_pure)],length(inv_data),1));


