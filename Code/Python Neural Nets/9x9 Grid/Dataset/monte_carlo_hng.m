% Monte  Carlo Heston Nandi
clearvars;
Maturity        = 10:30:250;%30:30:210  10:30:250
K               = 0.9:0.025:1.1;
S               = 1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';
num_sim = 100000;
z = normrnd(0,1,num_sim,max(Maturity));
z0 = normrnd(0,1,num_sim,1);
S0 = ones(num_sim,1);
Y = zeros(num_sim,max(Maturity));
h = zeros(num_sim,max(Maturity));
S = zeros(num_sim,max(Maturity));
load("id_dfc18d626cbb42f1_data_price_norm_205851clean.mat")
N = length(data_price);
omega_vec = data_price(:,4);
alpha_vec = data_price(:,1);
gamma_vec = data_price(:,3);
beta_vec = data_price(:,2);
h0_vec = data_price(:,5);
yield_curve = data_price(:,6:14);
price_vec = zeros(N,length(Maturity)*length(K));
for j = 1:300
    if mod(j,100)==0
        disp(j/N)
    end
    r = interp1(Maturity,yield_curve(j,:),1:250);
    r(isnan(r)) = 0;
    omega = omega_vec(j);
    alpha = alpha_vec(j);
    gamma = gamma_vec(j);
    beta  = beta_vec(j);
    h0 = h0_vec(j);
    for t = 1:max(Maturity)
        if t==1 
            h(:,t) = omega+alpha*(z0-gamma*sqrt(h0)).^2+beta*h0;
            Y(:,t) = r(t)/252-0.5*h(:,t)+sqrt(h(:,t)).*z(:,t); 
            S(:,t) = S0.*exp(Y(:,t));
        else
            h(:,t) = omega+alpha*(z(:,t-1)-gamma*sqrt(h(:,t-1))).^2+beta*h(:,t-1);
            Y(:,t) = r(t)/252-0.5*h(:,t)+sqrt(h(:,t)).*z(:,t);
            S(:,t) = S(:,t-1).*exp(Y(:,t));
        end
    end
    price_vec(j,:) = repmat(exp(-yield_curve(j,:).*Maturity/252),1,length(K)).*mean(max(repmat(S(:,Maturity),1,length(K))-repmat(K,1,length(Maturity)),zeros(num_sim,1)));
end
mape_mat = abs((data_price(:,15:end)-price_vec)./data_price(:,15:end));
mape = mean(mape_mat(1:300,:),1);