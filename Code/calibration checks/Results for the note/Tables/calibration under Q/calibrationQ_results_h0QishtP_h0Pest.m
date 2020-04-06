clear;
year_nums = {'2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'};
load('num_weeks');
num_allweeks = sum(num_weeks);
num_params = 4;
num_years = length(year_nums);
params_tmp = struct();
allparams = zeros(num_allweeks, num_params);
allsig20 = zeros(num_allweeks, num_params);
mean_params_year = zeros(num_years, num_params);
std_params_year = zeros(num_years, num_params);
mean_sig20_year = zeros(num_years, 1);
std_sig20_year = zeros(num_years, 1);
k = 1;
for cur_num = 1:num_years
    load(['params_options_', year_nums{cur_num}, '_h0ashtMLEP_MSE_interiorpoint_noYield.mat']);
    num_weeks_in_year = num_weeks(cur_num);
    year_data(cur_num).params_tmp = zeros(num_weeks_in_year, num_params);
    year_data(cur_num).MSE = zeros(num_weeks_in_year, 1);
    year_data(cur_num).IVRMSE = zeros(num_weeks_in_year, 1);
    year_data(cur_num).OptionsLikelihood = zeros(num_weeks_in_year, 1);
    year_data(cur_num).MAPE = zeros(num_weeks_in_year, 1);
    j = 1;
    year_params = zeros(num_weeks_in_year, num_params);
    year_sig20 = zeros(num_weeks_in_year, 1);
    for i=1:length(values)
        if ~isempty(values{1,i})
            year_params(j, :) = values{1,i}.hngparams;
            year_sig20(j, :) = values{1,i}.sig20;
            year_data(cur_num).params_tmp(j, :) = year_params(j, :);
            year_data(cur_num).MSE(j) = values{1,i}.MSE;
            year_data(cur_num).IVRMSE(j) = values{1,i}.IVRMSE;
            year_data(cur_num).OptionsLikelihood(j) = values{1,i}.optionsLikhng;
            year_data(cur_num).MAPE(j) = values{1,i}.MAPE;
            allparams(k, :) = year_params(j, :);
            allsig20(k, :) = year_sig20(j, :);
            j = j + 1;
            k = k + 1;
        end
    end
    mean_params_year(cur_num, :) = mean(year_params);
    std_params_year(cur_num, :) = std(year_params);
    mean_sig20_year(cur_num, :) = mean(year_sig20);
    std_sig20_year(cur_num, :) = std(year_sig20);
end
mean_MSE = arrayfun(@(x) mean(x.MSE), year_data);
mean_IVRMSE = arrayfun(@(x) nanmean(x.IVRMSE), year_data);
mean_OptionsLikelihood = arrayfun(@(x) mean(x.OptionsLikelihood), year_data);
mean_MAPE = arrayfun(@(x) mean(x.MAPE), year_data);


FID = fopen('calibrationQ_results_h0QishtP_h0Pest_calls_2010_2018.tex', 'w');
fprintf(FID, '%%&pdflatex \r%%&cont-en \r%%&pdftex \r');
fprintf(FID, '\\documentclass[10pt]{article} \n\\usepackage{latexsym,amsmath,amssymb,graphics,amscd} \n');
fprintf(FID, '\\usepackage{multirow} \n\\usepackage{booktabs} \n');
fprintf(FID, '\\usepackage{tabularx} \n\\usepackage[hang,footnotesize]{caption} \n');
fprintf(FID, '\\usepackage[pdftex]{graphicx} \n\\usepackage{color}\n\\textwidth15.8 cm\n\\textheight20.8 cm\n\\oddsidemargin.4cm\n\\evensidemargin.4cm \n\\begin{document} \n');

fprintf(FID, '\\noindent\\begin{center} Results are obtained with $h_0^P$ estimated \\end{center} \n');

fprintf(FID, '\\noindent\\makebox[\\textwidth]{ \n');
fprintf(FID, '\\begin{tabularx}{1.3\\textwidth}{X} \n \\scalebox{0.85}{ \n\\begin{tabular}{cccccccccc} \n');
fprintf(FID, '\\toprule \n');
fprintf(FID, '\\multicolumn{10}{c}{{\\bf CALIBRATED PARAMETERS ON WEDNESDAYS, $h_0^Q = h_t^P$}} \\\\\n');
fprintf(FID, '\\midrule \n');
fprintf(FID, '{$\\boldsymbol{\\theta}$}&{\\bf 2010}&{\\bf 2011}&{\\bf 2012}&{\\bf 2013}&{\\bf 2014}&{\\bf 2015}&{\\bf 2016}&{\\bf 2017}&{\\bf 2018}\\\\ \n');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 1;
fprintf(FID, ' { $\\omega$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 2;
fprintf(FID, ' { $\\alpha$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 3;
fprintf(FID, ' { $\\beta$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 4;
fprintf(FID, ' { $\\gamma^{*}$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_params_year(:, param_ind));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_params_year(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');

fprintf(FID, ' { $h_0^Q=h_t^P$ }& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_sig20_year(:)');
fprintf(FID, ' & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_sig20_year(:)');

fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');

fprintf(FID, ' { $MSE$ }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_MSE(:)');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { $IVRMSE$ }& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_IVRMSE(:)');

fprintf(FID, '\\bottomrule\n');
fprintf(FID, '\\end{tabular}}\n\\end{tabularx}}\n');

fprintf(FID, '\n  ');


fprintf(FID, '\\vspace{3 cm}\n');

fprintf(FID, '\n  ');



fprintf(FID, '\\end{document}');
fclose(FID);