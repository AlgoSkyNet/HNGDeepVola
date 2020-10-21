% Optimizer: ImpVola to HNG Parameters
clc;clearvars;
load("data_for_IV_moneyness.mat")
Maturity        = 10:30:250;
Moneyness       = 1.1:-0.025:0.9;
S               = 2000;
K               = S./Moneyness;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';

Ntest = 1000;%30878;
lb_base = [0,0,-1500,1e-12,1e-12];
ub_base = [1,1,1500,1000,1];
Nparameters = 5;
init = [1e-6,0.6,200,1e-7,1e-4];
scaler = magnitude(init);
for i =1:Ntest
    prices_vec = reshape(reshape(price(i,:,:),9,9)',81,1);
    r = reshape(repmat(rates(i,:),9,1),81,1);
    f_min_raw = @(params, scaler) (mean((price_Q_order(params.*scaler, data_vec, r./252)' - prices_vec).^2));    
    f_min = @(params) f_min_raw(params, scaler);
    nonlincon_fun = @(params) nonlincon_order(params, scaler);

    %parameter bounds, scaled
    lb = lb_base./scaler;
    ub = ub_base./scaler;
    init_scale = init./scaler;
    %optimization specs
    opt = optimoptions('fmincon', ...
                'Display', 'iter',...
                'Algorithm', 'interior-point',...
                'MaxIterations', 400,...
                'MaxFunctionEvaluations',2500, ...
                'TolFun', 1e-6,...
                'TolX', 1e-9,...
                'TypicalX',init./scaler);
    tic
    [xxval,fval,exitflag] = fmincon(f_min, init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    time_mat(i,:) = toc;
    opt_params_raw(i, :) = xxval;
    flag_mat(i,:) = exitflag;
    fval_mat(i,:)= fval;
    opt_params_clean(i, :) = opt_params_raw(i, :).*scaler;   
    
end
            