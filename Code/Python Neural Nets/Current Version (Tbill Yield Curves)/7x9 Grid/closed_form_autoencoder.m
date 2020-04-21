%% testing out of sample autoencoder performance with close form solution
clearvars,clc;%parpool(50);
load("forecasts.mat")

params_real = xdata(:,[4,1,2,3,5]);
N = length(params_real);
interest_rate = interestrate;
params =forecast_x_test(:,[4,1,2,3,5]);
%real_surface = reshape(permute(real_data,[1,3,2]),N,63);
real_surface = reshape(real_data,N,63);

Maturity        = 30:30:210;
K               = 0.9:0.025:1.1;
S               = 1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';
constraint = params(:,2).*params(:,4).^2+params(:,3);
vio_constraint = constraint>=1;
vio_clear = constraint <1;
bad_idx = [];
Nrun = N;
for i = 1:Nrun
    
    if mod(i,100)==0
        disp(i/N)
    end

    interestRates = interest_rate(i,:);
    for k = 1:length(interestRates)
        if interestRates(k)<0
            interestRates(k)=0;
        end
    end
    interestRates = repmat(interestRates,1,9)';
   % initialisation for first week
    autoencoder_price(i,:) = price_Q_clear(params(i, 1:4), data_vec, interestRates/252, params(i, 5));
    real_price(i,:) = price_Q_clear(params_real(i, 1:4), data_vec, interestRates/252, params_real(i, 5));
    autoencoder_vola(i,:)  = blsimpv(data_vec(:,3)',  data_vec(:,1)',  interestRates', data_vec(:,2)'/252, autoencoder_price(i,:)); 
    real_vola(i,:)= blsimpv(data_vec(:,3)',  data_vec(:,1)',  interestRates', data_vec(:,2)'/252, real_price(i,:)); 
    if any(isnan(real_vola(i,:))) || any(isnan(autoencoder_vola(i,:))) || any(isnan(real_price(i,:))) || any(isnan(autoencoder_price(i,:)))
        bad_idx(end+1) =i;
    elseif any(real_vola(i,:)==0) ||any(real_price(i,:)==0)
        bad_idx(end+1) =i;
    elseif any(real_vola(i,:)>1) ||any(real_price(i,:)>1)
        bad_idx(end+1) =i;
    end

    
end
%% Plots
% all
figure
sgtitle("xaxis = parameter,price,vola || yaxis = all,violation,no violation")
idx = setxor(1:Nrun,bad_idx);
autoencoder_vola_tmp = autoencoder_vola(idx,:);
real_vola_tmp = real_vola(idx,:);
autoencoder_price_tmp = autoencoder_price(idx,:);
real_price_tmp = real_price(idx,:);
params_tmp = params(idx,:);
params_real_tmp = params_real(idx,:);
rel_error= 100*abs((params_tmp-params_real_tmp)./params_real_tmp);
rel_error_price = 100*abs((autoencoder_price_tmp-real_price_tmp)./real_price_tmp);
rel_error_vola = 100*abs((autoencoder_vola_tmp-real_vola_tmp)./real_vola_tmp);
subplot(3,5,1)
boxplot(rel_error)
subplot(3,5,2)
heatmap(reshape(mean(rel_error_price),9,7));
subplot(3,5,3)
histogram(reshape(rel_error_price,prod(size(rel_error_price),1)))
subplot(3,5,4)
heatmap(reshape(mean(rel_error_vola),9,7));
subplot(3,5,5)
histogram(reshape(rel_error_vola,prod(size(rel_error_vola),1)))

%violation
idx = setxor(1:Nrun,bad_idx);
count = 1:Nrun;count = count(vio_constraint(1:Nrun));
idx = intersect(idx,count);
autoencoder_vola_tmp = autoencoder_vola(idx,:);
real_vola_tmp = real_vola(idx,:);
autoencoder_price_tmp = autoencoder_price(idx,:);
real_price_tmp = real_price(idx,:);
params_tmp = params(idx,:);
params_real_tmp = params_real(idx,:);
rel_error= 100*abs((params_tmp-params_real_tmp)./params_real_tmp);
rel_error_price = 100*abs((autoencoder_price_tmp-real_price_tmp)./real_price_tmp);
rel_error_vola = 100*abs((autoencoder_vola_tmp-real_vola_tmp)./real_vola_tmp);
subplot(3,5,6)
boxplot(rel_error)
subplot(3,5,7)
heatmap(reshape(mean(rel_error_price),9,7));
subplot(3,5,8)
histogram(reshape(rel_error_price,prod(size(rel_error_price),1)))
subplot(3,5,9)
heatmap(reshape(mean(rel_error_vola),9,7));
subplot(3,5,10)
histogram(reshape(rel_error_vola,prod(size(rel_error_vola),1)))


%no violation
idx = setxor(1:Nrun,bad_idx);
count = 1:Nrun;count = count(vio_clear(1:Nrun));
idx = intersect(idx,count);
autoencoder_vola_tmp = autoencoder_vola(idx,:);
real_vola_tmp = real_vola(idx,:);
autoencoder_price_tmp = autoencoder_price(idx,:);
real_price_tmp = real_price(idx,:);
params_tmp = params(idx,:);
params_real_tmp = params_real(idx,:);
rel_error= 100*abs((params_tmp-params_real_tmp)./params_real_tmp);
rel_error_price = 100*abs((autoencoder_price_tmp-real_price_tmp)./real_price_tmp);
rel_error_vola = 100*abs((autoencoder_vola_tmp-real_vola_tmp)./real_vola_tmp);
subplot(3,5,11)
boxplot(rel_error)
subplot(3,5,12)
heatmap(reshape(mean(rel_error_price),9,7));
subplot(3,5,13)
histogram(reshape(rel_error_price,prod(size(rel_error_price),1)))
subplot(3,5,14)
heatmap(reshape(mean(rel_error_vola),9,7));
subplot(3,5,15)
histogram(reshape(rel_error_vola,prod(size(rel_error_vola),1)))