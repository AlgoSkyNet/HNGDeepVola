%matlab_optimizer_mle_v2
clc; close all; clearvars;
%% Initialziation
omega = 1.8e-9;alpha = 1.5e-6;beta = 0.63;gamma = 250;lambda = 2.4;
sigma0=(alpha+omega)/(1-beta-alpha*gamma.^2);
Init = [omega,alpha,beta,gamma,lambda,sigma0];
sc_fac           =   magnitude(Init);
Init_scale_mat   =   Init./sc_fac;
Init_scale       =   Init_scale_mat(1,:);
scaler           =   sc_fac(1,:);  

r = 0.005/252;
lb_h0 = [1e-12,0,0,-1000,-100,1e-12]./sc_fac;
ub_h0 =[1,1,100,2000,100,1]./sc_fac;
A = [];
b = [];
Aeq = [];
beq = [];
%nonlincon contains nonlinear constraint
%% yearly data
data = load('SP500_data.txt');
dates = [data(:,2),week(datetime(data(:,1),'ConvertFrom','datenum'))];
year=2015;
win_len = 2520; %around 10years
num_week = max(dates(dates(:,1)==year,2));
opt_ll = NaN*ones(num_week,1);
params_mle_weekly=NaN*ones(num_week,6);
hist_vola = NaN*ones(num_week,1);
%% optimization
% TODO SCALE PARAMETERS AS IN CALLOPTI!!!!!
for i=1:num_week
    display(i);
    indicee = find(((dates(:,2)==i).*(dates(:,1)==year))==1,1,'first');
    if isempty(indicee)
        continue
    end
    logret_1y = data(indicee-252:indicee-1,4);
    logret = data(indicee-win_len:indicee-1,4);

    hist_vola(i) = sqrt(252)*std(logret);

    f_min_raw = @(par, scaler) ll_hng_n_h0(par.*scaler,logret,r);
    gs = GlobalSearch('XTolerance',1e-9,'StartPointsToRun','bounds-ineqs','Display','final');
    if i~=1 
        
        x0=[params;Init_scale_mat];
        scaler  = sc_fac; 
        fmin_ = zeros(2,1);
        xmin_ = zeros(2,6);
        
        f1      = f_min_raw(x0,scaler);
        x2      = params_mle_weekly(i-1,:);
        scaler  = scale_tmp;
        f2      = f_min_raw(x2,scaler);
         
            Init_scale = x2;
            scaler = scale_tmp;
        
        
        for j=1:2
            f_min = @(params) f_min_raw(params,scaler);
            problem = createOptimProblem('fmincon','x0',x0(j,:),...
                'objective',f_min,'lb',lb_h0,'ub',ub_h0,'nonlcon',@nonlincon);
            [xmin_(j,:),fmin_(j)] = run(gs,problem);
        end
        [fmin,idx] = min(fmin_);
        xmin = xmin_(idx,:);
%     opt = optimoptions('fmincon','Display','iter','Algorithm','interior-point','MaxIterations',1000,'MaxFunctionEvaluations',1500, 'TolFun',1e-4,'TolX',1e-4);
% for j=1:2
% [xmin_(j,:),fmin_(j)] = fmincon(f_min,x0(j,:),[],[],[],[],lb_h0,ub_h0,@nonlincon,opt);
% [fmin,idx] = min(fmin_);
%          xmin = xmin_(idx,:);
% end
    else
        f_min = @(params) f_min_raw(params,scaler); 

        gs = GlobalSearch('XTolerance',1e-9,...
            'StartPointsToRun','bounds-ineqs','NumTrialPoints',2e3);
        problem = createOptimProblem('fmincon','x0',Init_scale,...
                'objective',f_min,'lb',lb_h0,'ub',ub_h0,'nonlcon',@nonlincon);
        [xmin,fmin] = run(gs,problem);    
%     opt = optimoptions('fmincon','Display','iter','Algorithm','interior-point','MaxIterations',1000,'MaxFunctionEvaluations',1500, 'TolFun',1e-4,'TolX',1e-4);
% 
% [xmin,fmin] = fmincon(f_min,Init_scale,[],[],[],[],lb_h0,ub_h0,@nonlincon,opt);
    end    
    params = xmin;
    scale_tmp       =   scaler;
    params_original = xmin.*scaler;
    opt_ll(i)= fmin;
    params_mle_weekly(i,:)=xmin;
    params_mle_weekly_original(i,:)=xmin.*scaler;
end
%params_Q_mle_weekly = [params_mle_weekly(:,1),params_mle_weekly(:,2),params_mle_weekly(:,3),params_mle_weekly(:,4)+params_mle_weekly(:,5)+0.5];
params_Q_mle_weekly = [params_mle_weekly_original(:,1),params_mle_weekly_original(:,2),params_mle_weekly_original(:,3),params_mle_weekly_original(:,4)+params_mle_weekly_original(:,5)+0.5];
sig2_0 = params_mle_weekly_original(:,6);
save('weekly_2015_mle_opt.mat','sig2_0','hist_vola','params_Q_mle_weekly')