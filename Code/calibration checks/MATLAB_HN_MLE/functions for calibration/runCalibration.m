function [fValOut]=runCalibration(params, weeksprices, data, sig2_0, SP500_date_prices_returns_realizedvariance_interestRates, ...
    Dates,dataRet, vola_tmp, index, rValue)

%% weekly optimization
j = 1;
totalOLL = 0;
curWeeks = unique(weeksprices);
for i = curWeeks
    data_week = data(:,(weeksprices == i))';
    if isempty(data_week)
        disp(strcat('no data for week !'))
        continue
    end
    
    if j > 1
        logret = dataRet(index(1):index(j) - 1,4);
    end
   
    
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
        if interestRates(k)<0
            interestRates(k)=0;
        end
    end
    
    if nargin > 9 && rValue
        % use average r for the vola dynamics
        if rValue
            r = rValue/252;
        end
    else
        if j == 1
            r=max([interestRates',0])/252;
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
                b=0;
            end
        end
    end
     if j > 1
        [sigmaseries] = sim_hng_Q_n(params,logret,r,sig2_0);
    else
        sigmaseries = sig2_0;
     end
     j = j + 1;
    
    Price         =   data_week(:, 1)';
    %yields        =   interestRates;
    blsimpvVal       =   blsimpv(data_week(:, 4),  data_week(:, 3), r_cur, data_week(:, 2)/252, data_week(:, 1));
    indNaN = find(isnan(blsimpvVal));
    %num_NaN_implVols = length(indNaN);
    blsimpvVal(indNaN) = data_week(indNaN, 6);
    blsvegaVal = blsvega(data_week(:, 4),  data_week(:, 3), r_cur(:), data_week(:, 2)/252, blsimpvVal(:));
    
    hngPrice      =   abs(price_Q(params, data_week, r_cur./252, sigmaseries(end))) ;
    epsilonhng    =   (Price - hngPrice) ./  blsvegaVal';
    s_epsilon2hng       =   mean(epsilonhng(:).^2);
    numOptions    =  length(data_week(:, 1));
    %optionsLikhng    = -.5 * numOptions * (log(2 * pi) + log(s_epsilon2hng) + 1 + sum(log(blsvegaVal)) * 2/numOptions);
    optionsLikNorm    = - numOptions*log(s_epsilon2hng);
    %values{i}           =   struc;
    %totalOLL = totalOLL + optionsLikhng;
    totalOLL = totalOLL + optionsLikNorm;
    
end
fValOut = -totalOLL/length(curWeeks);

