% Optimizer: ImpVola to HNG Parameters
load("dataset.mat")
r = 0.005;
data_vec = [combvec(strikes,maturities);S0*ones(1,Nmaturities*Nstrikes)]';
%% Finding starting values:
% error_matlab = zeros(Ntest,Ntrain);
% init_params = zeros(Ntest,Nparameters);
% init_error = zeros(Ntest,1);
% for i = 1:Ntest
%     for j=1:Ntrain
%         error_matlab(i,j) = sum(sum((prediction(i,:,:)-y_train_trafo2(j,:,:)).^2));
%         if j==1
%             init_params(i,:) = X_train(j,:);
%             init_error(i) = error_matlab(i,j);
%         elseif error_matlab(i,j)<init_error(i)
%             init_params(i,:) = X_train(j,:);
%             init_error(i) = error_matlab(i,j);
%         end  
%     end
%     if mod(i,100)==0
%         disp(i)
%     end
% end
%% optimization
load("init_params.mat")
lb = [0,0,-1000,1e-12,1e-12];
ub = [1,1,1000,1000,2];
opti_params = zeros(Ntrain,Nparameters);
prediction_trafo = reshape(prediction,Ntest,Nmaturities*Nstrikes);
delete(gcp('nocreate'))
pool_  = parpool('HenrikHNG',64);
%gs = GlobalSearch('XTolerance',1e-9,'StartPointsToRun','bounds-ineqs','Display','final');
opt = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxIterations',20,'MaxFunctionEvaluations',150);    
for i = 1:Ntest
    x0 = init_params(i,:);
    f_min =@(params) fun2opti(params,prediction_trafo(i,:),r,data_vec);
    opti_params(i,:) = fmincon(f_min,x0,[],[],[],[],lb,ub,@nonlincon_nn,opt);
    %problem = createOptimProblem('fmincon','x0',x0,...
    %            'objective',f_min,'lb',lb,'ub',ub,'nonlcon',@nonlincon_nn);
    %[xmin,fmin] = run(gs,problem);    
    disp(i)
end
