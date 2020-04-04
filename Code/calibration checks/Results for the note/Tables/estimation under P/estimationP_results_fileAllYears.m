clear;
load('num_weeks.mat');
load('weekly_10to18_mle_opt_noh0est_check_rng.mat');
num_years = 9;
num_params = 5;
j = 1;
mean_vals = zeros(num_years, num_params + 1);
std_vals = zeros(num_years, num_params + 1);
for i = 1:num_years
    mean_vals(i, 1:num_params) = mean(params_P_mle_weekly(j:j + num_weeks(i) - 1, :));
    mean_vals(i, num_params + 1) = mean(sigma2_last(j:j + num_weeks(i) - 1, :));
    std_vals(i, 1:num_params) = std(params_P_mle_weekly(j:j + num_weeks(i) - 1, :));
    std_vals(i, num_params + 1) = std(sigma2_last(j:j + num_weeks(i) - 1, :));
    j = j + num_weeks(i);
end

FID = fopen('estimationP_results_calls_2010_2018.tex', 'w');
fprintf(FID, '%%&pdflatex \r%%&cont-en \r%%&pdftex \r');
fprintf(FID, '\\documentclass[10pt]{article} \n\\usepackage{latexsym,amsmath,amssymb,graphics,amscd} \n');
fprintf(FID, '\\usepackage{multirow} \n\\usepackage{booktabs} \n');
fprintf(FID, '\\usepackage{tabularx} \n\\usepackage[hang,footnotesize]{caption} \n');
fprintf(FID, '\\usepackage[pdftex]{graphicx} \n\\usepackage{color}\n\\textwidth15.8 cm\n\\textheight20.8 cm\n\\oddsidemargin.4cm\n\\evensidemargin.4cm \n\\begin{document} \n');

fprintf(FID, '\\noindent\\makebox[\\textwidth]{ \n');
fprintf(FID, '\\begin{tabularx}{1.3\\textwidth}{X} \n \\scalebox{0.85}{ \n\\begin{tabular}{cccccccccc} \n');
fprintf(FID, '\\toprule \n');
fprintf(FID, '\\multicolumn{10}{c}{{\\bf MLE ESTIMATES ON WEDNESDAYS (10 YEARS PRIOR)}} \\\\\n');
fprintf(FID, '\\midrule \n');
fprintf(FID, '{$\\boldsymbol{\\theta}$}&{\\bf 2010}&{\\bf 2011}&{\\bf 2012}&{\\bf 2013}&{\\bf 2014}&{\\bf 2015}&{\\bf 2016}&{\\bf 2017}&{\\bf 2018}\\\\ \n');
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 1;
fprintf(FID, ' { $\\omega$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_vals(:, param_ind));
fprintf(FID, ' & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_vals(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 2;
fprintf(FID, ' { $\\alpha$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_vals(:, param_ind));
fprintf(FID, ' & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_vals(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 3;
fprintf(FID, ' { $\\beta$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_vals(:, param_ind));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_vals(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 4;
fprintf(FID, ' { $\\gamma$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_vals(:, param_ind));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_vals(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 5;
fprintf(FID, ' { $\\lambda$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_vals(:, param_ind));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_vals(:, param_ind));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
param_ind = 6;
fprintf(FID, ' { $h_0$ {\\bf last}}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4f$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_vals(:, param_ind));
fprintf(FID, ' & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_vals(:, param_ind));
fprintf(FID, '\\bottomrule\n');
fprintf(FID, '\\end{tabular}}\n\\end{tabularx}}\n');

fprintf(FID, '\n  ');


fprintf(FID, '\\vspace{3 cm}\n');

fprintf(FID, '\n  ');



fprintf(FID, '\\end{document}');
fclose(FID);