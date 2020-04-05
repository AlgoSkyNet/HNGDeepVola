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


%tan trafo
%data2 = (2*(data_pure-(10e-8)*sign(data_pure-repmat(mean(data_pure),length(data_pure),1)))-repmat([max(data_pure)+min(data_pure)],length(data_pure),1))./(repmat([max(data_pure)-min(data_pure)],length(data_pure),1));
%data = atanh(data2);

% log trafo
data = [log(data_pure(:,1:3)),data_pure(:,4),data_pure(:,5:end)];

% pure for  normal uni
%data  = [data_pure(:,1:5),log((data_pure(:,6:end))+10e-8)];


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
            ind = i;
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

% pure (normal dist)
%inv_data   = [sample_trafo(:,1:5),exp(yields_inv)-10e-8];
%inv_data   = inv_scaler(normalize(inv_data),mean(data_pure),std(data_pure));


%log trafo 4
inv_data = [exp(sample_trafo(:,1:3)),sample_trafo(:,4),exp(sample_trafo(:,5)),yields_inv];
inv_data = inv_scaler(normalize(inv_data),mean(data_pure),std(data_pure));

% tan trafo 
%inv_data = inv_scaler(sample_trafo,mean_,std_);
%inv_data = tanh(inv_data);
%inv_data = 0.5*(inv_data.*(repmat([max(data_pure)-min(data_pure)],length(inv_data),1))+repmat([max(data_pure)+min(data_pure)],length(inv_data),1));

% tan trafo normalised
%inv_data = inv_scaler(sample_trafo,mean_,std_);
%inv_data = inv_scaler(normalize(tanh(inv_data)),mean(data2),std(data2));
%inv_data = 0.5*(inv_data.*(repmat([max(data_pure)-min(data_pure)],length(inv_data),1))+repmat([max(data_pure)+min(data_pure)],length(inv_data),1));


%uni
%inv_data = inv_scaler(sample_trafo,mean(data_pure),std(data_pure));
%yields_tmp = inv_data(:,6:end);
%inv_data  = [min([params,sig2_0])+(max([params,sig2_0])-min([params,sig2_0])).*rand(Nsim,5),yields_tmp];

%uni semiscaled
%inv_data = inv_scaler(sample_trafo,mean(data_pure),std(data_pure));
%yields_tmp = inv_data(:,6:end);
%inv_data  = [min([params,sig2_0])+(max([params,sig2_0])-min([params,sig2_0])).*rand(Nsim,5),yields_tmp];
%inv_data = inv_data./std(inv_data).*std(data_pure);

inv_data = [inv_data(:,1:5),max(inv_data(:,6:end),0)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% not in use anymore (due to performance or redundance)

%normal dist ( = pure)
%inv_data  = mvnrnd(mean([params,sig2_0]),cov([params,sig2_0]),Nsim);

% log 5 no retrafo (approx normal dist due to law of large numbers)
%inv_data = inv_scaler(normalize(sample_trafo),mean(data_pure),std(data_pure));

%uni scaled (approx normal dist due to law of large numbers)
%inv_data  = min([params,sig2_0])+(max([params,sig2_0])-min([params,sig2_0])).*rand(Nsim,5);
%inv_data = inv_scaler(normalize(sample_trafo),mean(data_pure),std(data_pure));

% log trafo
%inv_data = inv_scaler(sample_trafo,mean_,std_);
%inv_data = [exp(inv_data(:,1:3)),inv_data(:,4),exp(inv_data(:,5)),yields_inv];

% log traf 2
%inv_data = inv_scaler(sample_trafo,mean_,std_);
%inv_data = inv_scaler(normalize([exp(inv_data(:,1:3)),inv_data(:,4),exp(inv_data(:,5)),yields_inv]),mean(data_pure),std(data_pure));

% log trafo 3 (same as log trafo 4)
%%inv_data = [exp(sample_trafo(:,1:3)),sample_trafo(:,4),exp(sample_trafo(:,5)),yields_inv];
%inv_data = inv_scaler(inv_data,mean_,std_);
%inv_data = inv_scaler(normalize(inv_data),mean(data_pure),std(data_pure));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


stats = [mean(data_pure);mean(inv_data);std(data_pure);std(inv_data)];
disp(stats);

%%
j=0;
fail3= 0;
fail1 =0;
constraint = zeros(1,Nsim);
param_clean = zeros(Nsim,5);
yields_clean = zeros(Nsim,4);
for i = 1:Nsim
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
    j=j+1;
    constraint(j) = b+a*g^2;
    param_clean(j,:) = [w,a,b,g,sig];
    yields_clean(j,:) = inv_data(i,6:end);
end
disp([mean(param_clean),mean(yields_clean);std(param_clean),std(yields_clean)]);
constraint = constraint(1:j);
param_clean = param_clean(1:j,:);
yields_clean = yields_clean(1:j,:);
subplot(2,4,1),histogram(param_clean(:,2),'Normalization','probability');title('alpha')
%set(gca, 'XScale', 'log')
subplot(2,4,2),histogram(param_clean(:,3),'Normalization','probability');title('beta')
%set(gca, 'XScale', 'log')
subplot(2,4,3),histogram(param_clean(:,4),'Normalization','probability');title('gamma')
subplot(2,4,4),histogram(param_clean(:,1),'Normalization','probability');title('omega')
%set(gca, 'XScale', 'log')
subplot(2,4,5),histogram(param_clean(:,5),'Normalization','probability');title('sigma')
%set(gca, 'XScale', 'log')
subplot(2,4,6),histogram(constraint,'Normalization','probability');title('constraint');
subplot(2,4,7),boxplot(yields_clean);title('yields');
subplot(2,4,8),boxplot(yields);title('yields true');

figure
subplot(1,4,1),histogram(yields(:,2),'Normalization','probability');
subplot(1,4,2),histogram(yields(:,2),'Normalization','probability');
subplot(1,4,3),histogram(yields(:,2),'Normalization','probability');
subplot(1,4,4),histogram(yields(:,2),'Normalization','probability');
figure
subplot(1,4,1),histogram(yields_clean(:,2),'Normalization','probability');
%set(gca, 'XScale', 'log')
subplot(1,4,2),histogram(yields_clean(:,2),'Normalization','probability');
%set(gca, 'XScale', 'log')
subplot(1,4,3),histogram(yields_clean(:,2),'Normalization','probability');
%set(gca, 'XScale', 'log')
subplot(1,4,4),histogram(yields_clean(:,2),'Normalization','probability');
%set(gca, 'XScale', 'log')
%%
monotonic = @(data_set) sign(diff(data_set')');
y11 = mean(monotonic(yields_clean),'all');
y12 = mean(monotonic(yields),'all');
y21 = mean(monotonic(yields_clean));
y22 = mean(monotonic(yields));
