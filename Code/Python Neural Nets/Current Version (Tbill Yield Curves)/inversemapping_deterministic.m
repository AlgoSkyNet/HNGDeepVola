%HNG-Optimization under Q 
clearvars,clc,close all;
load("id_3283354135d44b67_data_price_norm_231046clean.mat");
params = data_price(:,[5,1,2,3,4]);
interest_rate = data_price(:,6:12);
prices = data_price(:,13:end);
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
Nrun = 10;
for i = 1:Nrun
    int = ran_idx(i);
    interestRates = interest_rate(int,:);
    for k = 1:length(interestRates)
        if interestRates(k)<0
            interestRates(k)=0;
        end
    end
    % Goal function
    interestRates = repmat(interestRates,1,9)';
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
            'MaxIterations', 30,...
            'MaxFunctionEvaluations',400, ...
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
end
params_analysed = params(ran_idx(1:Nrun),:);
save('determinisitc_inv.mat','opt_params_clean','params_analysed')
%%
load('determinisitc_inv.mat')
rel_error= 100*abs((opt_params_clean-params(ran_idx(1:10),:))./params(ran_idx(1:10),:));
mean_error = mean(rel_error)
boxplot(rel_error)
heatmap(rel_error);


