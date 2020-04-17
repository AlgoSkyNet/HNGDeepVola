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
cil_params_year = zeros(num_years, num_params);
ciu_params_year = zeros(num_years, num_params);
std_params_year = zeros(num_years, num_params);
mean_sig20_year = zeros(num_years, 1);
median_sig20_year = zeros(num_years, 1);
median_persist_year = zeros(num_years, 1);
mean_persist_year = zeros(num_years, 1);
std_persist_year = zeros(num_years, 1);
median_params_year = zeros(num_years, num_params);
std_sig20_year = zeros(num_years, 1);
cil_sig20_year = zeros(num_years, 1);
alpha = 1-0.95;
k = 1;
for cur_num = 1:num_years
    load(['data for tables/calibrated h0 update/OS OptLL/params_options_', year_nums{cur_num}, '_h0_calibr_Upd.mat']);
    num_weeks_in_year = num_weeks(cur_num);
    year_data(cur_num).params_tmp = zeros(num_weeks_in_year, num_params);
    year_data(cur_num).MSE = zeros(num_weeks_in_year, 1);
    year_data(cur_num).IVRMSE = zeros(num_weeks_in_year, 1);
    year_data(cur_num).OptionsLikelihood = zeros(num_weeks_in_year, 1);
    year_data(cur_num).MAPE = zeros(num_weeks_in_year, 1);
    j = 1;
    year_params = zeros(num_weeks_in_year, num_params);
    year_sig20 = zeros(num_weeks_in_year, 1);
    year_persist = zeros(num_weeks_in_year, 1);
    for i=1:length(valuesOS)
        if ~isempty(valuesOS{1,i})
            year_params(j, :) = valuesOS{1,i}.hngparams;
            year_sig20(j, :) = valuesOS{1,i}.sig20;
            year_data(cur_num).params_tmp(j, :) = year_params(j, :);
            year_data(cur_num).MSE(j) = valuesOS{1,i}.MSE;
            display(year_data(cur_num).MSE(j));
            display(year_params(j, 3));
            year_data(cur_num).IVRMSE(j) = valuesOS{1,i}.IVRMSE;
            year_data(cur_num).OptionsLikelihood(j) = valuesOS{1,i}.optionsLikhng;
            year_data(cur_num).MAPE(j) = valuesOS{1,i}.MAPE;
            year_data(cur_num).persist(j) = year_params(j, 3)+year_params(j, 2)*year_params(j, 4).^2;
            year_persist(j) = year_data(cur_num).persist(j);
            allparams(k, :) = year_params(j, :);
            allpersist(k, :) = year_persist(j);
            allsig20(k, :) = year_sig20(j, :);
            j = j + 1;
            k = k + 1;
        end
    end
    n = j-1;
    std_params_year(cur_num, :) = std(year_params);
    tval = tinv(1-alpha/2,n - 1);
    cil_params_year(cur_num, :) =  tval.* std_params_year(cur_num, :)/sqrt(n); 
    mean_params_year(cur_num, :) = mean(year_params);
    mean_sig20_year(cur_num, :) = mean(year_sig20);
    median_params_year(cur_num, :) = median(year_params);
    median_persist_year(cur_num, :) = median(year_persist);
    mean_persist_year(cur_num, :) = mean(year_persist);
    std_persist_year(cur_num, :) = std(year_persist);
    median_sig20_year(cur_num, :) = median(year_sig20);
    std_sig20_year(cur_num, :) = std(year_sig20);
    cil_sig20_year(cur_num, :) =  tval.* std_sig20_year(cur_num, :)/sqrt(n); 
end
mean_MSE = arrayfun(@(x) mean(x.MSE), year_data);
median_MSE = arrayfun(@(x) median(x.MSE), year_data);
mean_IVRMSE = arrayfun(@(x) nanmean(x.IVRMSE), year_data);
mean_OptionsLikelihood = arrayfun(@(x) mean(x.OptionsLikelihood), year_data);
mean_MAPE = arrayfun(@(x) mean(x.MAPE), year_data);


FID = fopen('calibrQ_results_h0calibrUpd_h0Pest_calls_OS_10_18.tex', 'w');
fprintf(FID, '%%&pdflatex \r%%&cont-en \r%%&pdftex \r');
fprintf(FID, '\\documentclass[10pt]{article} \n\\usepackage{latexsym,amsmath,amssymb,graphics,amscd} \n');
fprintf(FID, '\\usepackage{multirow} \n\\usepackage{booktabs} \n');
fprintf(FID, '\\usepackage{tabularx} \n\\usepackage[hang,footnotesize]{caption} \n');
fprintf(FID, '\\usepackage[pdftex]{graphicx} \n\\usepackage{color}\n\\textwidth15.8 cm\n\\textheight20.8 cm\n\\oddsidemargin.4cm\n\\evensidemargin.4cm \n\\begin{document} \n');

fprintf(FID, '\\noindent\\begin{center} Results are obtained with $h_0^P$ estimated \\end{center} \n');

fprintf(FID, '\\noindent\\makebox[\\textwidth]{ \n');
fprintf(FID, '\\begin{tabularx}{1.3\\textwidth}{X} \n \\scalebox{0.7}{ \n\\begin{tabular}{cccccccccc} \n');
fprintf(FID, '\\toprule \n');
fprintf(FID, '\\multicolumn{10}{c}{{\\bf CALIBRATED PARAMETERS ON WEDNESDAYS, $h_0^Q$ IS LAST MLEP, THEN 1 WEEK UPDATED}} \\\\\n');
fprintf(FID, '\\midrule \n');
fprintf(FID, '{$\\boldsymbol{\\theta}$}&{\\bf 2010}&{\\bf 2011}&{\\bf 2012}&{\\bf 2013}&{\\bf 2014}&{\\bf 2015}&{\\bf 2016}&{\\bf 2017}&{\\bf 2018}\\\\ \n');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 1;
fprintf(FID, ' { $\\omega$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' {{\\bf std}}& $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_params_year(:, param_ind));
%fprintf(FID, ' {{\\bf ci}}& $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$& $(\\pm%.4e)$& $(\\pm%.4e)$ \\\\\n', cil_params_year(:, param_ind));
fprintf(FID, ' { {\\bf median}}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', median_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 2;
fprintf(FID, ' { $\\alpha$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' {{\\bf std}}& $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_params_year(:, param_ind));
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$& $(\\pm%.4e)$& $(\\pm%.4e)$ \\\\\n', cil_params_year(:, param_ind));
fprintf(FID, ' { {\\bf median}}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', median_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 3;
fprintf(FID, ' { $\\beta$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' {{\\bf std}}& $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_params_year(:, param_ind));
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$& $(\\pm%.4f)$& $(\\pm%.4f)$ \\\\\n', cil_params_year(:, param_ind));
fprintf(FID, ' { {\\bf median}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', median_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 4;
fprintf(FID, ' { $\\gamma^{*}$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' {{\\bf std}}& $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_params_year(:, param_ind));
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$& $(\\pm%.4f)$& $(\\pm%.4f)$ \\\\\n', cil_params_year(:, param_ind));
fprintf(FID, ' { {\\bf median}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', median_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');

fprintf(FID, ' { $h_0^Q$ }& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4f$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_sig20_year(:)');
fprintf(FID, ' {{\\bf std}}& $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_sig20_year(:)');
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$ & $(\\pm%.4e)$& $(\\pm%.4e)$& $(\\pm%.4e)$ \\\\\n', cil_sig20_year(:)');
fprintf(FID, ' { {\\bf median} }& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', median_sig20_year(:)');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');

fprintf(FID, ' { {\\bf persistency}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_persist_year(:)');
fprintf(FID, ' {{\\bf std}}& $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_persist_year(:)');
%fprintf(FID, ' {\\bf ci}& $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$ & $(\\pm%.4f)$& $(\\pm%.4f)$& $(\\pm%.4f)$ \\\\\n', cil_params_year(:, param_ind));
fprintf(FID, ' { {\\bf median}}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', median_persist_year(:)');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');

fprintf(FID, ' { {\\bf MSE} }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_MSE(:)');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { {\\bf median MSE} }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', median_MSE(:)');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { {\\bf IVRMSE} }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_IVRMSE(:)');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { {\\bf MAPE} }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_MAPE(:)');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { {\\bf OptLL} }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_OptionsLikelihood(:)');

fprintf(FID, '\\bottomrule\n');
fprintf(FID, '\\end{tabular}}\n\\end{tabularx}}\n');

fprintf(FID, '\n  ');


fprintf(FID, '\\vspace{3 cm}\n');

fprintf(FID, '\n  ');



fprintf(FID, '\\end{document}');
fclose(FID);