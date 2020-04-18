%HNG-Optimization under Q 
clearvars,clc,close all;
load("id_3283354135d44b67_data_price_norm_231046clean.mat");
load("id_3283354135d44b67_data_vola_norm_231046clean.mat");
params = data_price(:,[5,1,2,3,4]);
interest_rate = data_price(:,6:12);
prices = data_price(:,13:end);
volas = data_vola(:,13:end);
% Initialization  
Init = [1e-5,1e-6,0.7,300,1e-7];
scaler           =   magnitude(Init);
Init_scale  =   Init./scaler;
lb_mat           =   [1e-12, 0, 0, -1500,1e-12];
ub_mat           =   [1, 1, 1, 1500,1];  
N = length(data_price);
ran_idx = randperm(N);
Maturity        = 30:30:210;
K               = 0.9:0.025:1.1;
S               = 1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';

%%
Nrun = 100;
for i = 1:Nrun
    int = ran_idx(i);
    interestRates = interest_rate(int,:);
    for k = 1:length(interestRates)
        if interestRates(k)<0
            interestRates(k)=0;
        end
    end
    interestRates = repmat(interestRates,1,9)';
    % Goal function
    f_min_raw = @(params, scaler) (mean((price_Q_clear(params(1:4).*scaler(1:4), data_vec, interestRates/252,params(5).*scaler(5)) - prices(int,:)).^2));

    
    %% Algorithm 
    f_min = @(params) f_min_raw(params, scaler);

    % constraint,scaled
    nonlincon_fun = @(params) nonlincon_scale_v2(params, scaler);
    %parameter bounds, scaled
    lb = lb_mat./scaler;
    ub = ub_mat./scaler; 
    %optimization specs
    opt = optimoptions('fmincon', ...
            'Display', 'iter',...
            'Algorithm', "interior-point",...
            'MaxIterations',100,...
            'MaxFunctionEvaluations',1000, ...
            'TolFun', 1e-6,...
            'TolX', 1e-6,...
            'TypicalX',Init_scale);

            
    %local optimization
    [xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    % initialisation for first week
    opt_params_raw(i, :) = xxval;
    flags(i,:) = exitflag;
    goalval(i,:) = fval;
    opt_params_clean(i, :) = opt_params_raw(i, :).*scaler;   
    autoencoder_price(i,:) = price_Q_clear(opt_params_clean(i, 1:4), data_vec, interestRates/252, opt_params_clean(i, 5));
    autoencoder_vola(i,:)  = blsimpv(data_vec(:,3)',  data_vec(:,1)',  interestRates', data_vec(:,2)'/252, autoencoder_price(i,:)); 
end
params_analysed = params(ran_idx(1:Nrun),:);
prices_analysed = prices(ran_idx(1:Nrun),:);
volas_analysed  = volas(ran_idx(1:Nrun),:);
save('determinisitc_inv.mat','opt_params_clean','params_analysed','autoencoder_price','autoencoder_vola','volas_analysed','prices_analysed')
%%
load('determinisitc_inv.mat')
rel_error= 100*abs((opt_params_clean-params_analysed)./params_analysed);
rel_error_vola = 100*abs((autoencoder_vola-volas_analysed)./volas_analysed);
rel_error_prices = 100*abs((autoencoder_price-prices_analysed)./prices_analysed);
mean_error = mean(rel_error);
figure
boxplot(rel_error)
figure
heatmap(reshape(mean(rel_error_prices),9,7));
figure
heatmap(reshape(mean(rel_error_vola),9,7));


