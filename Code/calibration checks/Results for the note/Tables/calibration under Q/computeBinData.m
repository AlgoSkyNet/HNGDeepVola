%HNG-Optimization under Q
%Options: path = '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
clc;
clearvars;
close all;
warning('on')

%parpool()
%path                = 'C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
path                =  '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/Data/Datasets';
%path                =  'C://GIT/HenrikAlexJP/Data/Datasets';
stock_ind           = 'SP500';
year                = 2011;
useYield            = 0; % uses tbils now
useRealVola         = 0; % alwas use realized vola
useMLEPh0           = 1; % use last h_t from MLE under P as h0
algorithm           = 'interior-point';% 'sqp'
goal                =  'MSE'; % 'MSE';   'MAPE';  ,'OptLL';
path_               = strcat(path, '/', stock_ind, '/', 'Calls', num2str(year), '.mat');
load(path_);


bound                   = [100, 100];
formatIn                = 'dd-mmm-yyyy';

% start from the first Wednesday of 2015 and finish with the last Wednesday of 2015

DateString_start        = strcat('01-January-',num2str(year));
DateString_end          = strcat('31-December-',num2str(year));
date_start              = datenum(DateString_start, formatIn);
date_end                = datenum(DateString_end, formatIn);
wednessdays             = (weekday(date_start:date_end)==4);
Dates                   = date_start:date_end;
Dates                   = Dates(wednessdays);

% initialize with the data from MLE estimation for each week
%load(strcat('C:/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Code/Calibration MLE/','weekly_',num2str(year),'_mle_opt.mat'));
%load(strcat('C:/Users/TEMP/Documents/GIT/HenrikAlexJP/Code/calibration checks/MATLAB_HN_MLE/MLE_P estimation results/','weekly_',num2str(year),'_mle_opt.mat'));
%load(strcat('C:/GIT/HenrikAlexJP/Code/calibration checks/Calibration MLE P/Results with estimated h0P/','weekly_',num2str(year),'_mle_opt_h0est.mat'));

%load(strcat('data for tables/results calibration h0asUncondVMLEP esth0P/','params_options_',num2str(year),'_h0asUncondVarMLEP_MSE_interiorpoint_noYield.mat'));
%load(strcat('data for tables/results calibration hoashtMLEP esthoP/','params_options_',num2str(year),'_h0ashtMLEP_MSE_interiorpoint_noYield.mat'));
load(strcat('data for tables/results calibration h0Calibrated esth0P/','params_options_',num2str(year),'_h0_calibrated_MSE_interiorpoint_noYield.mat'));

if useRealVola || useMLEPh0
    num_params = 4;
else
    num_params = 5;
end


% bounds for maturity, moneyness, volumes, interest rates
Type                    = 'call';
MinimumVolume           = 100;
MinimumOpenInterest     = 100;
IfCleanNans             = 1;
TimeToMaturityInterval  = [8, 250];
MoneynessInterval       = [0.9, 1.1];

[OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptions(Dates, Type, ...
    TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans,...
    TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume, ...
    OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
    TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);

weeksprices             = week(datetime([OptionsStruct.date], 'ConvertFrom', 'datenum'));

idxj  = 1:length(unique(weeksprices));



data = [OptionsStruct.price; OptionsStruct.maturity; OptionsStruct.strike; OptionsStruct.priceunderlying; OptionsStruct.vega; OptionsStruct.implied_volatility];
% save('generaldata2015.mat', 'data', 'DatesClean', 'OptionsStruct', 'OptFeatures', 'idx');
MaturitiesBounds = [8 30 80 180 250];
MoneynessBounds = [.9 .95 .975 1 1.025 1.05 1.1];
MaturitiesBounds = [7 30 80 180 250];
MoneynessBounds = [.9-1e-9 .95 .975 1 1.025 1.05 1.1];
NumberOfContracts = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);
NumberOfContracts_all = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);
numMaturitiesBounds = length(MaturitiesBounds)-1;
MSE_bins = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);
MSE_bins_all = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);
mse_total = zeros(length(unique(weeksprices)),1);
mse_total_struct = 0;
num_options_all = 0;
mse_total_struct1 = 0;
ind = 1;
for i = unique(weeksprices)
    data_week = data(:,(weeksprices == i))';
    if isempty(data_week)
        disp(strcat('no data for week ',num2str(i),' in ',num2str(year),'!'))
        continue
    end
    num_options = values{1,i}.numOptions;
    num_options_all = num_options_all + num_options;
    for j = 1:num_options
        for k = 2:numMaturitiesBounds + 1
                if data_week(j, 2)> MaturitiesBounds(k - 1) && data_week(j, 2)<= MaturitiesBounds(k)
                    numMatint(j) = k - 1;
                end
        end
        mon_opt = data_week(j, 4)/data_week(j, 3);
        for k = 2:length(MoneynessBounds)
            if mon_opt> MoneynessBounds(k - 1) && mon_opt <= MoneynessBounds(k)
                    numMonint(j) = k - 1;
            end 
        end
        MSE_bins(numMatint(j),numMonint(j)) = MSE_bins(numMatint(j),numMonint(j)) + (values{1,i}.hngPrice(j) - values{1,i}.Price(j)).^2;
        mse_total(ind) = mse_total(ind) + (values{1,i}.hngPrice(j) - values{1,i}.Price(j)).^2;
        
        NumberOfContracts(numMatint(j),numMonint(j)) = NumberOfContracts(numMatint(j),numMonint(j)) + 1;
    end
    mse_total_struct1 = mse_total_struct1 + mse_total(ind)/num_options;
    ind = ind + 1;
    mse_total_struct = mse_total_struct + values{1,i}.MSE;
    %MSE_bins_all = MSE_bins_all + MSE_bins;
    %NumberOfContracts_all = NumberOfContracts_all + NumberOfContracts;
end
%MSE_bins_all = MSE_bins_all./NumberOfContracts_all;
MSE_bins_all = MSE_bins./NumberOfContracts;



%FID = fopen('Prices_h0QisUncondVar_h0Pest_2018_MSE.tex', 'w');
FID = fopen('Prices_h0Qcalibrated_h0Pest_2011_MSE.tex', 'w');
fprintf(FID, '%%&pdflatex \r%%&cont-en \r%%&pdftex \r');
fprintf(FID, '\\documentclass[10pt]{article} \n\\usepackage{latexsym,amsmath,amssymb,graphics,amscd} \n');
fprintf(FID, '\\usepackage{multirow} \n\\usepackage{booktabs} \n');
fprintf(FID, '\\usepackage{tabularx} \n\\usepackage[hang,footnotesize]{caption} \n');
fprintf(FID, '\\usepackage[pdftex]{graphicx} \n\\usepackage{color}\n\\textwidth15.8 cm\n\\textheight20.8 cm\n\\oddsidemargin.4cm\n\\evensidemargin.4cm \n\\begin{document} \n');

fprintf(FID, '\\noindent\\makebox[\\textwidth]{ \n');
fprintf(FID, '\\begin{tabularx}{1.3\\textwidth}{X} \n \\scalebox{0.85}{ \n\\begin{tabular}{llccccccc} \n');
fprintf(FID, '\\toprule \n');
%fprintf(FID, '\\multicolumn{9}{c}{{\\bf IN-SAMPLE PRICING ERRORS (MSE), $h_0^Q = \\dfrac{\\omega_0 + \\alpha_0}{1-\\beta_0 - \\alpha_0 \\gamma_0^{*2}}$, WITH $\\omega_0, \\alpha_0, \\beta_0, \\gamma_0^{*2}$ FROM MLE UNDER P }} \\\\\n');
%fprintf(FID, '\\multicolumn{9}{c}{{\\bf IN-SAMPLE PRICING ERRORS (MSE), $h_0^Q = h_t^P$ }} \\\\\n');
fprintf(FID, '\\multicolumn{9}{c}{{\\bf IN-SAMPLE PRICING ERRORS (MSE), $h_0^Q$ CALIBRATED }} \\\\\n');
fprintf(FID, '\\midrule \n');
fprintf(FID, '\\multicolumn{2}{c}{ }&\\multicolumn{6}{c}{{\\bf Moneyness $S _0/K $}}&\\multicolumn{1}{c}{{\\bf Across}}\\\\ \n');
fprintf(FID, '\\cmidrule(r){3-9} \n');

fprintf(FID, ' &{\\bf Maturities}');
fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 0.9, 0.95);
fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 0.95, 0.975);
fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 0.975, 1);
fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 1, 1.025);
fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 1.025, 1.05);
fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 1.05, 1.1);
fprintf(FID, '&\\multicolumn{1}{c}{{\\bf Moneyness}}');

fprintf(FID, '\\\\\n');


fprintf(FID, '\\cmidrule(r){1-9} \n');
fprintf(FID, '\\multirow{4}{*}{\\bf In-Sample Error}');
fprintf(FID, '&\\multirow{1}{*}{\\makebox[2cm]{$ {\\bf %d}\\leq T < {\\bf %d}$}} \n', 8, 30);
for ii=1:6
     fprintf(FID, '&%4.3f',MSE_bins_all(1, ii));
end
fprintf(FID, '&%4.3f',nansum(MSE_bins_all(1, :).*NumberOfContracts(1, :))/nansum(NumberOfContracts(1, :)));
fprintf(FID, '\\\\\n  ');

fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T < {\\bf %d}$} \n', 30, 80);
for ii=1:6
     fprintf(FID, '&%4.3f',MSE_bins_all(2, ii));
end
fprintf(FID, '&%4.3f',nansum(MSE_bins_all(2, :).*NumberOfContracts(2, :))/nansum(NumberOfContracts(2, :)));
fprintf(FID, '\\\\\n  ');

fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T < {\\bf %d}$} \n', 80, 180);
for ii=1:6
     fprintf(FID, '&%4.3f',MSE_bins_all(3, ii));
end
fprintf(FID, '&%4.3f',nansum(MSE_bins_all(3, :).*NumberOfContracts(3, :))/nansum(NumberOfContracts(3, :)));

fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T \\leq {\\bf %d}$} \n', 180, 250);
for ii=1:6
     fprintf(FID, '&%4.3f',MSE_bins_all(4, ii));
end
fprintf(FID, '&%4.3f',nansum(MSE_bins_all(4, :).*NumberOfContracts(4, :))/nansum(NumberOfContracts(4, :)));
fprintf(FID, '\\\\\n  ');

fprintf(FID, '\\midrule \n');
fprintf(FID, '\\midrule \n');

fprintf(FID, '{\\bf Across Maturities}&  ');
for ii=1:6
    temp(ii) = nansum(MSE_bins_all(:, ii).* NumberOfContracts(:, ii));
    fprintf(FID, '&%4.3f',temp(ii)/nansum(NumberOfContracts(:, ii)));
end
fprintf(FID, '&%4.3f',nansum(temp(:))/nansum(NumberOfContracts(:)));

fprintf(FID, '\\\\\n  ');
fprintf(FID, '\\midrule \n');
fprintf(FID, '\\midrule \n');



fprintf(FID, '\\\\\n  ');
fprintf(FID, '\\midrule \n');


fprintf(FID, '\\bottomrule\n');
fprintf(FID, '\\end{tabular}}\n\\end{tabularx}}\n');

fprintf(FID, '\\end{document}');
fclose(FID);
