
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%           THIS FILE NEEDS MATLAB VERSION 2018a OR HIGHER TO RUN         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars; clc;close all;

%% Initialisation

% Configuration of underlying data
years     = [2010:2014,2016];%:2018;
goals     = ["MSE"];%,"MAPE","OptLL"];
path_data = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/Calibration Calloption/';
path_data_old = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/Calibration Calloption/old results b4 change/weird results/';


%% Concentate underlying Data
alldata = {};alldata_old = {};
k = 0;
for y = years
    for goal = goals
        k = k+1;
        file       = strcat(path_data,'params_v2_options_',num2str(y),'_h0asRealVola_',goal,'_interiorpoint_noYield.mat');
        tmp        = load(file);
        alldata{k} = tmp.values;
        file       = strcat(path_data_old,'params_Options_',num2str(y),'_h0asRealVola_',goal,'_InteriorPoint_noYield.mat');
        tmp        = load(file);
        alldata_old{k} = tmp.values;
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
        flag(Ninputs)     = alldata{1,j}{1,m}.optispecs.flag;  
        mse_old(Ninputs,:)    = alldata_old{1,j}{1,m}.MSE;
        mape_old(Ninputs,:)   = alldata_old{1,j}{1,m}.MAPE;
        params_old(Ninputs,:) = alldata_old{1,j}{1,m}.hngparams;
        sig2_0_old(Ninputs)   = alldata_old{1,j}{1,m}.sig20; 
        yields_old(Ninputs,:) = alldata_old{1,j}{1,m}.yields;
        num_opt(Ninputs,:)  =  alldata{1,j}{1,m}.numOptions;  
        num_opt_old(Ninputs,:)= alldata_old{1,j}{1,m}.numOptions;  
    end
end
sig2_0 = sig2_0';
sig2_0_old = sig2_0';
%%
corr_  = corr(params)
mean_ = mean(params)
median_ = median(params)
corr_size_err = corr(mse,num_opt)
corr_beta_err =corr(mse,params(:,3))

figure
histogram(mse)

corr_old  = corr(params_old)
mean_old = mean(params_old)
median_old = median(params_old)
corr_size_err_old = corr(mse_old,num_opt_old)
corr_beta_err_old =corr(mse,params(:,3))

figure
histogram(mse_old)

corr_beta_err =corr(f_vec,param_vec(:,3));
corr_  = corr(param_vec);
mean_ = mean(param_vec);
median_ = median(param_vec);
corr_size_err = corr(f_vec',num_vec');
figure
histogram(f_vec)
corr_beta_err =corr(f_vec,param_vec(:,3));
%%
%STATISTICS
figure("Name",num2str(2010))
j =0;
bad_idx = (f_vec>quantile(f_vec,0.8));
bad_weeks =unique(weeksprices);
bad_weeks = bad_weeks(bad_idx);
weeksplot=bad_weeks; %unique(weekprices);
for i=weeksplot
    j = j+1;
    data_week = data(:,(weeksprices == i))';
    k = ceil(sqrt(length(weeksplot)));
    subplot(k,k,j)
    [xq,yq] = meshgrid(unique(data_week(:,2)),unique(data_week(:,3)));
    vq = griddata(data_week(:,2),data_week(:,3),data_week(:,1),xq,yq);  %(x,y,v) being your original data for plotting points
    surf(xq,yq,vq)
    hold on
    scatter3(data_week(:,2),data_week(:,3),data_week(:,1),"o","k",'filled');
    ylim([1000 1300])
    xlim([0 250])
    view(0,90)
    colormap(jet(256));
    caxis([0, 100]);
    colorbar
    title(strcat("week ",num2str(i)))
end

