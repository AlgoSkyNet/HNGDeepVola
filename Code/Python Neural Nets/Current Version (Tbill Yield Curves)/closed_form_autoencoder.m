%% testing out of sample autoencoder performance with close form solution
load("dataset")
load("true_dataset")
params = data(:,[5,1,2,3,4]);
interest_rate = data(:,6:12);
forecast = data(:,13:end);
params_real = data_real(:,[5,1,2,3,4]);
real_surface = data_real(:,13:end);
N = length(data);
Maturity        = 30:30:210;
K               = 0.9:0.025:1.1;
S               = 1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';

for i = 1:N
    interestRates = interest_rate(i,:);
    for k = 1:length(interestRates)
        if interestRates(k)<0
            interestRates(k)=0;
        end
    end
    interestRates = repmat(interestRates,1,9)';
   % initialisation for first week
    autoencoder_price(i,:) = price_Q_clear(params(i, 1:4), data_vec, interestRates/252, params(i, 5));
    autoencoder_vola(i,:)  = blsimpv(data_vec(:,3)',  data_vec(:,1)',  interestRates', data_vec(:,2)'/252, autoencoder_price(i,:)); 
end
rel_error= 100*abs((params-params_real)./params_real);
rel_error_vola = 100*abs((autoencoder_vola-real_surface)./real_surface);
mean_error = mean(rel_error);
figure
boxplot(rel_error)
figure
heatmap(reshape(mean(rel_error_prices),9,7));
figure
heatmap(reshape(mean(rel_error_vola),9,7));