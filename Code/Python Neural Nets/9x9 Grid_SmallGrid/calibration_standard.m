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
mape_total = 100*abs((opt_params_clean-param(1:236,:))./param(1:236,:));
pers = param(1:236,1).*param(1:236,3).^2+param(1:236,2);
pers_pred = opt_params_clean(:,1).*opt_params_clean(:,3).^2+opt_params_clean(:,2);
mape_total(:,6) = 100*abs((pers-pers_pred)./pers);
%mape = mape_total(flag_mat==1,:);
mape  = mape_total;
summary = [quantile(mape,0.05);quantile(mape,0.25);quantile(mape,0.5);mean(mape);quantile(mape,0.75);quantile(mape,0.95);max(mape)];


for i=1:236
    disp(i);
    r = reshape(repmat(rates(i,:),9,1),81,1);
    price_forecast(i,:) = price_Q_order(opt_params_clean(i,:), data_vec, r./252);
    prices_vecc(i,:) = reshape(reshape(price(i,:,:),9,9)',81,1)';
    mape_price(i,:)= 100*abs((price_forecast(i,:)-prices_vecc(i,:))./prices_vecc(i,:));
end
summary_price = [quantile(mape_price,0.05);quantile(mape_price,0.25);quantile(mape_price,0.5);mean(mape_price);quantile(mape_price,0.75);quantile(mape_price,0.95);max(mape_price)];
figure
heatmap([1.1:-0.025:0.9],[10:30:250],reshape(summary_price(4,:),9,9)','ColorLimits',[0 150])

xlabel("Moneyness")
ylabel("Maturity")
load("data_calib_moneyness.mat")
for i=1:58826
    if mid(i,100)==0
        disp(i);
    end
    r = reshape(repmat(rates_calib(i,:),9,1),81,1);
    price_calib(i,:) = price_Q_order(params_calib(i,:), data_vec, r./252);
    prices_vecc_calib(i,:) = reshape(reshape(price_calibtrue(i,:,:),9,9)',81,1)';
    mape_price(i,:)= 100*abs((price_calib(i,:)-prices_vecc_calib(i,:))./prices_vecc_calib(i,:));
end
summary_price_calib = [quantile(mape_price,0.05);quantile(mape_price,0.25);quantile(mape_price,0.5);mean(mape_price);quantile(mape_price,0.75);quantile(mape_price,0.95);max(mape_price)];

figure
heatmap([1.1:-0.025:0.9],[10:30:250],reshape(summary_price_calib(4,:),9,9)','ColorLimits',[0 150])
xlabel("Moneyness")
ylabel("Maturity")



%% Out-of-sample 
Maturity        = 10:30:250;
Moneyness2       = 1.2:-0.020:0.8;
S               = 2000;
K               = S./Moneyness2;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';
for i=1:236
    disp(i);
    r = reshape(repmat(rates(i,:),length(Moneyness2),1),[],1);
    price_forecast2(i,:) = price_Q_order(opt_params_clean(i,:), data_vec, r./252);
    price2(i,:) = price_Q_order(param(i,:), data_vec, r./252); 
    mape_price2(i,:)= 100*abs((price_forecast2(i,:)-price2(i,:))./price2(i,:));
end
summary_price_calib2 = [quantile(mape_price2,0.05);quantile(mape_price2,0.25);quantile(mape_price2,0.5);mean(mape_price2);quantile(mape_price2,0.75);quantile(mape_price2,0.95);max(mape_price2)];


figure
heatmap(Moneyness2,[10:30:250],reshape(summary_price_calib2(4,:),[],9)','ColorLimits',[0 150])
xlabel("Moneyness")
ylabel("Maturity")


for i=1:58826
    if mod(i,100)==0
        disp(i);
    end
    r = reshape(repmat(rates_calib(i,:),length(Moneyness2),1),[],1);
    price_calib2(i,:) = price_Q_order(params_calib(i,:), data_vec, r./252);
    price_c2(i,:) = price_Q_order(param_true(i,:), data_vec, r./252); 
    mape_price3(i,:)= 100*abs((price_calib2(i,:)-price_c2(i,:))./price_c2(i,:));
end
summary_price_calib3 = [quantile(mape_price3,0.05);quantile(mape_price3,0.25);quantile(mape_price3,0.5);mean(mape_price3);quantile(mape_price3,0.75);quantile(mape_price3,0.95);max(mape_price3)];

figure
heatmap(Moneyness2,[10:30:250],reshape(summary_price_calib3(4,:),[],9)','ColorLimits',[0 150])
xlabel("Moneyness")
ylabel("Maturity")
