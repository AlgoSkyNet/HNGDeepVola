function [fValOut]=runCalibration(params, weeksprices, data, sig2_0, SP500_date_prices_returns_realizedvariance_interestRates, ...
    Dates,dataRet, vola_tmp, index, rValue)

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
    if dataRet(index(j),1) ~= Dates(j)
        disp('problem with dates!');
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
    
    if nargin > 8 && rValue
        % use average r for the vola dynamics
        if rValue
            r = rValue/252;
        end
    else
        if j == 1
            r=max([interestRates',0])/252;
        end
    end
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
    struc.blsimpv       =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, data_week(:, 1));
    indNaN = find(isnan(struc.blsimpv));
    struc.num_NaN_implVols = length(indNaN);
    struc.blsimpv(indNaN) = data_week(indNaN, 6);
    struc.blsvega = blsvega(data_week(:, 4),  data_week(:, 3), r_cur(:), data_week(:, 2)/252, struc.blsimpv(:));
    struc.Price         =   data_week(:, 1)';
    struc.hngPrice      =   abs(price_Q(params, data_week, r_cur./252, sigmaseries(end))) ;
    struc.epsilonhng    =   (struc.Price - struc.hngPrice) ./  struc.blsvega';
    s_epsilon2hng       =   mean(struc.epsilonhng(:).^2);
    struc.optionsLikhng    = -.5 * struc.numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1 + sum(log(struc.blsvega)) * 2/struc.numOptions);
    
    values{i}           =   struc;
    totalOLL = totalOLL + struc.optionsLikhng;
    
end
fValOut = -totalOLL/length(values);

