function [fValOut, values]=getCalibratedDatah0(params, weeksprices, data, SP500_date_prices_returns_realizedvariance_interestRates, Dates,dataRet, vola_tmp, index,rValue)

%% weekly optimization
j = 1;
totalOLL = 0;
for i = unique(weeksprices)
    data_week = data(:,(weeksprices == i))';
    if isempty(data_week)
        disp(strcat('no data for week !'))
        continue
    end
    
    if j > 1
        logret = dataRet(index(1):index(j) - 1,4);
    end
    
    
    struc               =  struct();
    struc.numOptions    =  length(data_week(:, 1));
    
    % compute interest rates for the weekly options
    interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:end, ...
        SP500_date_prices_returns_realizedvariance_interestRates(1,:) ==  dataRet(index(j),1));
    if isempty(interestRates)
        interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:end, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) ==  dataRet(index(j),1)-1);
    end
    if all(isnan(interestRates))
        interestRates = SP500_date_prices_returns_realizedvariance_interestRates(5:end, ...
            SP500_date_prices_returns_realizedvariance_interestRates(1,:) ==  dataRet(index(j),1)-1);
    end
    for k = 1:length(interestRates)
        if interestRates(k) < 0
            interestRates(k) = 0;
        end
    end
    if nargin > 8 && rValue
        % use average r for the vola dynamics
        if rValue
            r = rValue/252;
        end
    else
        if j == 1
            r = max([interestRates',0])/252;
        end
    end
    
    r_cur = zeros(length(data_week), 1);
    for k = 1:length(data_week)
        if data_week(k, 2) < 21 && ~isnan(interestRates(1))
            r_cur(k) = interestRates(1);
        else
            notNaN = ~isnan(interestRates);
           if length(interestRates) == 4
                daylengths = [21, 63, 126, 252]./252;
            else
                daylengths = [21, 42, 13*5, 126, 252]./252;
            end
            r_cur(k) = interp1(daylengths(notNaN), interestRates(notNaN), data_week(k, 2)./252);
            if isnan(r_cur(k))
                disp('interest rate problem');
            end
        end
    end
    if j > 1
        [sigmaseries] = sim_hng_Q_n(params(1:4),logret,r,params(5));
    else
        sigmaseries = params(5);
    end
     j = j + 1;
    
    struc.Price         =   data_week(:, 1)';
    struc.Date = dataRet(index(j-1),1);
    struc.DateStr = datestr(dataRet(index(j-1),1));
    struc.yields        =   interestRates;
    struc.blsPrice      =   blsprice(data_week(:, 4), data_week(:, 3), r_cur, data_week(:, 2)/252, vola_tmp(i), 0)';
    struc.blsimpv       =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, data_week(:, 1));
    indNaN = find(isnan(struc.blsimpv));
    struc.num_NaN_implVols = length(indNaN);
    struc.blsimpv(indNaN) = data_week(indNaN, 6);
    struc.blsvega = blsvega(data_week(:, 4),  data_week(:, 3), r_cur(:), data_week(:, 2)/252, struc.blsimpv(:));
    
    struc.hngPrice      =   abs(price_Q(params, data_week, r_cur./252, sigmaseries(end))) ;
    struc.epsilonhng    =   (struc.Price - struc.hngPrice) ./  struc.blsvega';
    s_epsilon2hng       =   mean(struc.epsilonhng(:).^2);
    struc.optionsLikhng    = -.5 * struc.numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1 + sum(log(struc.blsvega)) * 2/struc.numOptions);
    struc.optionsLikNorm    = - log(sum(((struc.hngPrice - struc.Price).^2)./(struc.blsvega'.^2))/struc.numOptions);
    struc.sig20         =   sigmaseries(end);
    struc.blsimpvhng    =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, struc.hngPrice');
    struc.epsilonbls    =   (struc.Price - struc.blsPrice) ./ data_week(:,5)';
    s_epsilon2bls       =   mean(struc.epsilonbls(:).^2);
    struc.optionsLikbls    = -.5 * struc.numOptions * (log(2 * pi) + log(s_epsilon2bls) + 1 + sum(log(struc.blsvega)) * 2/struc.numOptions);
    
    struc.meanPrice     =   mean(data_week(:, 1));
    struc.hngparams     =   params;
    struc.countneg      =   sum(struc.hngPrice <= 0);
    struc.matr          =   [struc.Price; struc.hngPrice; struc.blsPrice];
    struc.maxAbsEr      =   max(abs(struc.hngPrice - struc.Price));
    struc.IVRMSE        =   sqrt(mean(100 * (struc.blsimpv - struc.blsimpvhng).^2));
    struc.MAPE          =   mean(abs(struc.hngPrice - struc.Price)./struc.Price);
    struc.MaxAPE        =   max(abs(struc.hngPrice - struc.Price)./struc.Price);
    struc.MSE           =   mean((struc.hngPrice - struc.Price).^2);
    struc.RMSE          =   sqrt(struc.MSE);
    struc.RMSEbls       =   sqrt(mean((struc.blsPrice - struc.Price).^2));
    struc.sigma2series   = sigmaseries;
    values{i}           =   struc;
    %totalOLL = totalOLL + struc.optionsLikhng;
    totalOLL = totalOLL + struc.optionsLikNorm;
    
end
fValOut = -totalOLL/length(values);

