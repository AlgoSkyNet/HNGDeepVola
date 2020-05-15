function [fValOut]=runCalibration(params, weeksprices, data, sig2_0, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index)
       
%% weekly optimization
j = 1;
totalOLL = 0;
for i = unique(weeksprices)
    data_week = data(:,(weeksprices == i))';
    if isempty(data_week)
        disp(strcat('no data for week !'))
        continue
    end
    
    if (j - 1)
     logret = dataRet(index(1):index(j),4);
    else
        logret = dataRet(index(j),4);
    end
    
    struc               =  struct();
    struc.numOptions    =  length(data_week(:, 1));
    % compute interest rates for the weekly options
        interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:9, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j));
        if isempty(interestRates)
            interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:9, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-1);
        end
        if all(isnan(interestRates))
            interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:9, ...
                SP500_date_prices_returns_realizedvariance_interestRates(1,:) == Dates(j)-1);
        end
    for k = 1:length(interestRates)
        if interestRates(k)<0
            interestRates(k)=0;
        end
    end
    r=max([interestRates',0])/252;
    j = j + 1;
    r_cur = zeros(length(data_week), 1);
        for k = 1:length(data_week)
            if data_week(k, 2) < 21 && ~isnan(interestRates(1))
                r_cur(k) = interestRates(1);
            else
                notNaN = ~isnan(interestRates);
                daylengths = [21, 42, 13*5, 126, 252]./252;
                r_cur(k) = interp1(daylengths(notNaN), interestRates(notNaN), data_week(k, 2)./252);
                if isnan(r_cur(k))
                    b=0;
                end
            end
        end
        [~, sigmaseries] = ll_hng_Q_n(params,logret,r,sig2_0);
        
        
    struc.Price         =   data_week(:, 1)';
    struc.yields        =   interestRates;
    struc.blsPrice      =   blsprice(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, vola_tmp(i), 0)';
    struc.blsimpv       =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, data_week(:, 1));
    indNaN = find(isnan(struc.blsimpv));
    struc.num_NaN_implVols = length(indNaN);
    struc.blsimpv(indNaN) = data_week(indNaN, 6);
    struc.blsvega = blsvega(data_week(:, 4),  data_week(:, 3), r_cur(:), data_week(:, 2)/252, struc.blsimpv(:));
    
%     f_min_raw = @(params,scaler,h0) ((log(mean(((price_Q(params.*scaler, data_week, r_cur./252, h0)'-data_week(:, 1))./struc.blsvega).^2))));
% 
%         f_min = @(params) f_min_raw(params(1:num_params), scaler, sig2_0(i));
%     % constraint,scaled
%     nonlincon_fun = @(params) nonlincon_scale_v2(params, scaler);
%     %parameter bounds, scaled
%     lb = lb_mat./scaler;
%     ub = ub_mat./scaler; 
%     %optimization specs
%         opt = optimoptions('fmincon', ...
%                 'Display', 'final',...
%                 'Algorithm', algorithm,...
%                 'MaxIterations', 3000,...
%                 'MaxFunctionEvaluations',20000, ...
%                 'TolFun', 1e-6,...
%                 'TolX', 1e-9,...
%                 'TypicalX', Init(i,:)./scaler);
%              
%     struc.optispecs = struct();
%     struc.optispecs.optiopt = opt;
% 
%     %local optimization
%     [xxval,fval,exitflag] = fmincon(f_min, Init_scale, [], [], [], [], lb, ub, nonlincon_fun, opt);
    % initialisation for first week
    
    %struc.optispecs.flag = exitflag;
    %struc.optispecs.goalval = fval;
    opt_params_clean(i, :) = params;   
    scale_tmp           =   magnitude(opt_params_clean(i, :));
    
    struc.hngPrice      =   abs(price_Q(opt_params_clean(i,:), data_week, r_cur./252, sigmaseries(end))) ;
    struc.epsilonhng    =   (struc.Price - struc.hngPrice) ./  struc.blsvega';
    s_epsilon2hng       =   mean(struc.epsilonhng(:).^2);
    struc.optionsLikhng =   -.5 * struc.numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1);
    values{i}           =   struc;
    totalOLL = totalOLL + struc.optionsLikhng;
    
end 
fValOut = -totalOLL/length(values);

