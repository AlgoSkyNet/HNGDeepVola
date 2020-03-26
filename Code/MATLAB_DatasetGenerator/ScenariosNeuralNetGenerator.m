% Dataset Generator
% This Program generates the dataset for our neural net.

clearvars; clc;close all;

%% Initialisation
path_data = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/final option calibration/';
alldata = {};
k=1;
%concentate all datasets.
for y=2010:2018
    for goal = ["MSE","MAPE","OptLL"]
        file = strcat(path_data,'params_Options_',num2str(y),'_h0asRealVola_',goal,'_InteriorPoint_noYield.mat');
        alldata{k} = load(file);
        k = k+1;
    end
end

%rng('default')
N               = length(values);
Maturity        = 30:30:210;
K               = 0.9:0.025:1.1;
S               = 1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);

data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';


%% DATASET GENERATION: CURRENTLY UNDER CONSTRUCTION
Nsim            = 50000;
scenario_data   = zeros(Nsim, 7+Nstrikes*Nmaturities);
fail1           = 0;
fail2           = 0;
params          = compare(:,2,:);
min_            = reshape(min(params),1,4);
max_            = reshape(max(params),1,4);
Sig             = [min(sig2_0),max(sig2_0)];
fprintf('%s','Generatiting Prices. Progress: 0%')
for i = 1:Nsim
    price = -ones(1,Nmaturities*Nstrikes);
    while any(any(price < 0)) || any(any(price' > exp(-r/252*data_vec(:,2)/252).*data_vec(:,1))) %check for violation of intrinsiv bounds
        a = 1;
        b = 1;
        g = 1;
        % ToDo: Optimize random draw to fit distribution better!!!
        % (especially distribution of b+a*g^2)
        fail2 = fail2-1;
        while (b+a*g^2 >= 1)
            a = min_(2)+(max_(2)-min_(2)).*rand(1,1);
            b = min_(3)+(max_(3)-min_(3)).*rand(1,1);
            g = min_(4)+(max_(4)-min_(4)).*rand(1,1);
            fail2 = fail2 +1;
        end
        w       = min_(1)+(max_(1)-min_(1)).*rand(1,1);
        %Sig_    = (w+a)/(1-a*g^2-b);
        Sig_    = Sig(1)+(Sig(2)-Sig(1)).*rand(1,1);
        price   = price_Q_clear([w,a,b,g],data_vec,r/252,Sig_);
        fail1   = fail1+1;
    end
    fail1 = fail1-1;
    if ismember(i,floor(Nsim*[1/(5*log10(Nsim*100)):1/(5*log10(Nsim*100)):1]))
        fprintf('%0.5g',round(i/(Nsim)*100,1)),fprintf('%s',"%"),fprintf('\n')
        fprintf('Number of fails'),disp([fail1,fail2])
    end
    scenario_data(i,:) = [a, b, g, w, Sig_,(a+w)/(1-a*g^2-b), b+a*g^2, price];%reshape(price', [1,Nstrikes*Nmaturities])];  
end
data_price = scenario_data;
price_vec  = zeros(1,Nmaturities*Nstrikes);
bad_idx    = [];
fprintf('%s','Calculating Imp Volas. Progress: 0%')
for i = 1:Nsim
    if ismember(i,floor(Nsim*[1/(5*log10(Nsim*100)):1/(5*log10(Nsim*100)):1]))
        fprintf('%0.5g',round(i/(Nsim)*100,1)),fprintf('%s',"%"),fprintf('\n')
    end
    price_vec = data_price(i,8:end);
    vola(i,:) = blsimpv_vec(data_vec,r,price_vec);
    if any(isnan(vola(i,:))) || any(vola(i,:)==0)
        bad_idx(end+1) = i;
    end
end 
idx               = setxor(1:Nsim,bad_idx);
data_vola         = data_price(:,1:7);
data_vola(:,8:70) = vola;
data_vola         = data_vola(idx,:);
save(strcat('data_price_','maxbounds','_',num2str(length(data_price)),'_',num2str(r),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity)),'.mat'),'data_price')
save(strcat('data_vola_','maxbounds','_',num2str(length(data_price)),'_',num2str(r),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity)),'.mat'),'data_vola')



%% VISUALISATION FOR CONTROL PURPOSES
% Summary
    fprintf('\n')
    disp(strcat("rate/number of pos. price-data: ",num2str((sum(sum((scenario_data(:,8:end)>=0))))/(length(scenario_data)*Nmaturities*Nstrikes)*100),"% , ",num2str((sum(sum((scenario_data(:,8:end)>=0)))))));
    disp(['max price: ',    num2str(max(max(scenario_data(:,8:end))))])
    disp(['min price: ',    num2str(min(min(scenario_data(:,8:end))))])
    disp(['mean price: ',   num2str(mean(mean(scenario_data(:,8:end))))])
    disp(['median price: ', num2str(median(median(scenario_data(:,8:end))))])
    disp(['median alpha: ', num2str(median(scenario_data(:,1)))])
    disp(['median beta: ',  num2str(median(scenario_data(:,2)))])
    disp(['median gamma: ', num2str(median(scenario_data(:,3)))])
    disp(['median omega: ', num2str(median(scenario_data(:,4)))])
    disp(['median sigma: ', num2str(median(scenario_data(:,5)))])
    disp(['median stationary constraint: ', num2str(median(scenario_data(:,7)))])
figure
subplot(2,3,1),hist(scenario_data(:,1));title('alpha')
subplot(2,3,2),hist(scenario_data(:,2));title('beta')
subplot(2,3,3),hist(scenario_data(:,3));title('gamma')
subplot(2,3,4),hist(scenario_data(:,4));title('omega')
subplot(2,3,5),hist(scenario_data(:,5));title('sigma')
subplot(2,3,6),hist(scenario_data(:,7));title('constraint: b+a*g^2')
    
% Example plot
[X,Y]=meshgrid(K,Maturity);
surf(X',Y',reshape(data_vola(1,8:end),9,7));hold on;
scatter3(data_vec(:,1),data_vec(:,2),scenario_data(1,8:end));