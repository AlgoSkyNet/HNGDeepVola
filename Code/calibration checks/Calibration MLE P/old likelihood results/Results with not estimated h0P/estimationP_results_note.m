clear;
year_nums = {'2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'};
load('num_weeks');
num_allweeks = sum(num_weeks);
num_params = 5;
num_years = length(year_nums);
params_tmp = struct();
allparams = zeros(num_allweeks, num_params);
allsig20 = zeros(num_allweeks, num_params);
mean_params_year = zeros(num_years, num_params);
median_params_year = zeros(num_years, num_params);
std_params_year = zeros(num_years, num_params);
mean_sig20_year = zeros(num_years, 1);
median_sig20_year = zeros(num_years, 1);
cil_params_year = zeros(num_years, num_params);
cil_sig20_year = zeros(num_years, 1);
std_sig20_year = zeros(num_years, 1);
median_persist_year = zeros(num_years, 1);
mean_persist_year = zeros(num_years, 1);
mean_logLikVal_year = zeros(num_years, 1);
std_persist_year = zeros(num_years, 1);
alpha = 1-0.95;
k = 1;
for cur_num = 1:num_years
    %load(['data for tables/Results with estimated h0P/weekly_', year_nums{cur_num}, '_mle_opt_h0est.mat']);
    %load(['data for tables/Results with not estimated h0P/weekly_', year_nums{cur_num}, '_mle_opt_noh0est.mat']);
   % load(['data for tables/Results with estimated h0p rAv/weekly_', year_nums{cur_num}, '_mle_opt_h0est_rAv.mat']);
    %load(['data for tables/Results with estimated h0p and r/weekly_', year_nums{cur_num}, '_mle_opt_h0est_rest.mat']);
     load(['weekly_', year_nums{cur_num}, '_mle_opt_noh0est.mat']);
    num_weeks_in_year = num_weeks(cur_num);
    year_data(cur_num).params_tmp = zeros(num_weeks_in_year, num_params);
%     year_data(cur_num).MSE = zeros(num_weeks_in_year, 1);
%     year_data(cur_num).IVRMSE = zeros(num_weeks_in_year, 1);
%     year_data(cur_num).OptionsLikelihood = zeros(num_weeks_in_year, 1);
%     year_data(cur_num).MAPE = zeros(num_weeks_in_year, 1);
    j = 1;
    year_params = zeros(num_weeks_in_year, num_params);
    year_sig20 = zeros(num_weeks_in_year, 1);
     year_logLikVal = zeros(num_weeks_in_year, 1);
    year_persist = zeros(num_weeks_in_year, 1);
    
    for i=1:length(params_tmp_P)
        if sum(params_tmp_P(i,:))
            year_params(j, :) = params_tmp_P(i,:);
            year_sig20(j, :) = sig0_tmp(i);
            year_data(cur_num).params_tmp(j, :) = year_params(j, :);
            year_data(cur_num).persist(j) = year_params(j, 3)+year_params(j, 2)*year_params(j, 4).^2;
            year_data(cur_num).logLikVal(j) = logLikVals(i);
%             year_data(cur_num).MSE(j) = values{1,i}.MSE;
%             year_data(cur_num).IVRMSE(j) = values{1,i}.IVRMSE;
%             year_data(cur_num).OptionsLikelihood(j) = values{1,i}.optionsLikhng;
%             year_data(cur_num).MAPE(j) = values{1,i}.MAPE;
            year_persist(j) = year_data(cur_num).persist(j);
            year_logLikVal(j) = year_data(cur_num).logLikVal(j);
            allparams(k, :) = year_params(j, :);
            allsig20(k, :) = year_sig20(j, :);
            j = j + 1;
            k = k + 1;
        end
    end
    n = j - 1;
    if n~=length(find(params_tmp_P(:,1))) || num_weeks_in_year ~= n
        disp('problem');
    end
    std_params_year(cur_num, :) = std(year_params);
    tval = tinv(1-alpha/2,n - 1);
    cil_params_year(cur_num, :) =  tval.* std_params_year(cur_num, :)/sqrt(n); 
    mean_params_year(cur_num, :) = mean(year_params);
    mean_sig20_year(cur_num, :) = mean(year_sig20);
    median_params_year(cur_num, :) = median(year_params);
    median_sig20_year(cur_num, :) = median(year_sig20);
    median_persist_year(cur_num, :) = median(year_persist);
    mean_logLikVal_year(cur_num, :) = mean(year_logLikVal);
    mean_persist_year(cur_num, :) = mean(year_persist);
    std_persist_year(cur_num, :) = std(year_persist);
    std_sig20_year(cur_num, :) = std(year_sig20);
    cil_sig20_year(cur_num, :) =  tval.* std_sig20_year(cur_num, :)/sqrt(n); 
end
% mean_MSE = arrayfun(@(x) mean(x.MSE), year_data);
% mean_IVRMSE = arrayfun(@(x) nanmean(x.IVRMSE), year_data);
% mean_OptionsLikelihood = arrayfun(@(x) mean(x.OptionsLikelihood), year_data);
% mean_MAPE = arrayfun(@(x) mean(x.MAPE), year_data);


%FID = fopen('estMLEP_results_h0Pest_calls_10_18.tex', 'w');
%FID = fopen('estMLEP_results_h0Pest_calls_10_18.tex', 'w');
FID = fopen('estMLEP_results_noh0Pest_rAv_calls_10_18.tex', 'w');
%FID = fopen('estMLEP_results_h0Pest_rAv_calls_10_18.tex', 'w');
%FID = fopen('estMLEP_results_h0Pest_calls_10_18_check.tex', 'w');
%fprintf(FID, '%%&pdflatex \r%%&cont-en \r%%&pdftex \r');
fprintf(FID, '\\documentclass[10pt]{article} \n\\usepackage{latexsym,amsmath,amssymb,graphics,amscd} \n');
fprintf(FID, '\\usepackage{multirow} \n\\usepackage{booktabs} \n');
fprintf(FID, '\\usepackage{tabularx} \n\\usepackage[hang,footnotesize]{caption} \n');
fprintf(FID, '\\usepackage[pdftex]{graphicx} \n\\usepackage{color}\n\\textwidth15.8 cm\n\\textheight20.8 cm\n\\oddsidemargin.4cm\n\\evensidemargin.4cm \n\\begin{document} \n');

%fprintf(FID, '\\noindent\\begin{center} Results are obtained with $h_0^P$  estimated and $r$ estimated\\end{center} \n');
fprintf(FID, '\\noindent\\begin{center} Results are obtained with $h_0^P$ not estimated\\end{center} \n');
%fprintf(FID, '\\noindent\\begin{center} Results are obtained with $h_0^P$  estimated\\end{center} \n');

fprintf(FID, '\\noindent\\makebox[\\textwidth]{ \n');
fprintf(FID, '\\begin{tabularx}{1.3\\textwidth}{X} \n \\scalebox{0.7}{ \n\\begin{tabular}{cccccccccc} \n');
fprintf(FID, '\\toprule \n');
%fprintf(FID, '\\multicolumn{10}{c}{{\\bf ESTIMATED PARAMETERS ON WEDNESDAYS MLE UNDER P (10 YEARS), $h_0^P$ AND $r$ ESTIMATED}} \\\\\n');
fprintf(FID, '\\multicolumn{10}{c}{{\\bf ESTIMATED PARAMETERS ON WEDNESDAYS MLE UNDER P (10 YEARS), $h_0^P$ IS NOT ESTIMATED}} \\\\\n');
%fprintf(FID, '\\multicolumn{10}{c}{{\\bf ESTIMATED PARAMETERS ON WEDNESDAYS MLE UNDER P (10 YEARS), $h_0^P$ IS  ESTIMATED}} \\\\\n');
fprintf(FID, '\\midrule \n');
fprintf(FID, '{$\\boldsymbol{\\theta}$}&{\\bf 2010}&{\\bf 2011}&{\\bf 2012}&{\\bf 2013}&{\\bf 2014}&{\\bf 2015}&{\\bf 2016}&{\\bf 2017}&{\\bf 2018}\\\\ \n');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 1;
fprintf(FID, ' { $\\omega$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' {{\\bf std}}& $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_params_year(:, param_ind));
%fprintf(FID, ' {{\\bf ci}}& $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$& $(\\pm%.4e)$& $(\\pm%.4e)$ \\\\\n', cil_params_year(:, param_ind));
%fprintf(FID, ' { {\\bf median}}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', median_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 2;
fprintf(FID, ' { $\\alpha$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' {{\\bf std}}& $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_params_year(:, param_ind));
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$& $(\\pm%.4e)$& $(\\pm%.4e)$ \\\\\n', cil_params_year(:, param_ind));
%fprintf(FID, ' { {\\bf median}}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', median_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 3;
fprintf(FID, ' { $\\beta$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' {{\\bf std}}& $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_params_year(:, param_ind));
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$& $(\\pm%.4f)$& $(\\pm%.4f)$ \\\\\n', cil_params_year(:, param_ind));
%fprintf(FID, ' { {\\bf median}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', median_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 4;
fprintf(FID, ' { $\\gamma$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' {{\\bf std}}& $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_params_year(:, param_ind));
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$& $(\\pm%.4f)$& $(\\pm%.4f)$ \\\\\n', cil_params_year(:, param_ind));
%fprintf(FID, ' { {\\bf median}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', median_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 5;
fprintf(FID, ' { $\\lambda$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' {{\\bf std}}& $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_params_year(:, param_ind));
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$& $(\\pm%.4f)$& $(\\pm%.4f)$ \\\\\n', cil_params_year(:, param_ind));
%fprintf(FID, ' { {\\bf median}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', median_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');

fprintf(FID, ' { $h_0^P$ }& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_sig20_year(:)');
fprintf(FID, ' {{\\bf std}}& $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_sig20_year(:)');
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$& $(\\pm%.4e)$& $(\\pm%.4e)$ \\\\\n', cil_sig20_year(:)');
%fprintf(FID, ' { {\\bf median} }& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', median_sig20_year(:)');

fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');

fprintf(FID, ' { {\\bf persistency}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_persist_year(:)');
fprintf(FID, ' {{\\bf std}}& $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_persist_year(:)');
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$& $(\\pm%.4f)$& $(\\pm%.4f)$ \\\\\n', cil_params_year(:, param_ind));
%fprintf(FID, ' { {\\bf median}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', median_persist_year(:)');
%fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');

fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');

fprintf(FID, ' { {\\bf logLikValue}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_logLikVal_year(:)./2520');

% fprintf(FID, ' { {\\bf MSE} }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_MSE(:)');
% fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
% fprintf(FID, ' { {\\bf IVRMSE} }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_IVRMSE(:)');
% fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
% fprintf(FID, ' { {\\bf MAPE} }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_MAPE(:)');
% fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
% fprintf(FID, ' { {\\bf OptLL} }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_OptionsLikelihood(:)');

fprintf(FID, '\\bottomrule\n');
fprintf(FID, '\\end{tabular}}\n\\end{tabularx}}\n');

fprintf(FID, '\n  ');


fprintf(FID, '\\vspace{3 cm}\n');

fprintf(FID, '\n  ');



fprintf(FID, '\\end{document}');
fclose(FID);