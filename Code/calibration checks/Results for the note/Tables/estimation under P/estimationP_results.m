clear;
num_weeks = 0;
load('weekly_2010_mle_opt.mat');
num_weeks = num_weeks + length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2010 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2010 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2011_mle_opt.mat')
num_weeks = num_weeks + length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2011 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2011 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2012_mle_opt.mat')
num_weeks = num_weeks + length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2012 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2012 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2013_mle_opt.mat')
num_weeks = num_weeks + length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2013 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2013 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2014_mle_opt.mat')
num_weeks = num_weeks + length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2014 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2014 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2015_mle_opt.mat')
num_weeks = num_weeks + length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2015 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2015 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2016_mle_opt.mat')
num_weeks = num_weeks + length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2016 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2016 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2017_mle_opt.mat')
num_weeks = num_weeks + length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2017 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2017 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2018_mle_opt.mat')
num_weeks = num_weeks + length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2018 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2018 = std(params_tmp(params_tmp(:,1)~=0, :));

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
fprintf(FID, ' { $\\omega$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_vals_2010(1,1), mean_vals_2011(1,1),...
    mean_vals_2012(1,1), mean_vals_2013(1,1), mean_vals_2014(1,1), mean_vals_2015(1,1), mean_vals_2016(1,1), mean_vals_2017(1,1), mean_vals_2018(1,1));
fprintf(FID, ' & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_vals_2010(1,1), std_vals_2011(1,1),...
    std_vals_2012(1,1), std_vals_2013(1,1), std_vals_2014(1,1), std_vals_2015(1,1), std_vals_2016(1,1), std_vals_2017(1,1), std_vals_2018(1,1));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { $\\alpha$}& $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$ & $%.4e$& $%.4e$& $%.4e$ \\\\\n', mean_vals_2010(1,2), mean_vals_2011(1,2),...
    mean_vals_2012(1,2), mean_vals_2013(1,2), mean_vals_2014(1,2), mean_vals_2015(1,2), mean_vals_2016(1,2), mean_vals_2017(1,2), mean_vals_2018(1,2));
fprintf(FID, ' & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$ & $(%.4e)$& $(%.4e)$& $(%.4e)$ \\\\\n', std_vals_2010(1,2), std_vals_2011(1,2),...
    std_vals_2012(1,2), std_vals_2013(1,2), std_vals_2014(1,2), std_vals_2015(1,2), std_vals_2016(1,2), std_vals_2017(1,2), std_vals_2018(1,2));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { $\\beta$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_vals_2010(1,3), mean_vals_2011(1,3),...
    mean_vals_2012(1,3), mean_vals_2013(1,3), mean_vals_2014(1,3), mean_vals_2015(1,3), mean_vals_2016(1,3), mean_vals_2017(1,3), mean_vals_2018(1,3));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_vals_2010(1,3), std_vals_2011(1,3),...
    std_vals_2012(1,3), std_vals_2013(1,3), std_vals_2014(1,3), std_vals_2015(1,3), std_vals_2016(1,3), std_vals_2017(1,3), std_vals_2018(1,3));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { $\\gamma$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_vals_2010(1,4), mean_vals_2011(1,4),...
    mean_vals_2012(1,4), mean_vals_2013(1,4), mean_vals_2014(1,4), mean_vals_2015(1,4), mean_vals_2016(1,4), mean_vals_2017(1,4), mean_vals_2018(1,4));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_vals_2010(1,4), std_vals_2011(1,4),...
    std_vals_2012(1,4), std_vals_2013(1,4), std_vals_2014(1,4), std_vals_2015(1,4), std_vals_2016(1,4), std_vals_2017(1,4), std_vals_2018(1,4));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { $\\lambda$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_vals_2010(1,2), mean_vals_2011(1,2),...
    mean_vals_2012(1,2), mean_vals_2013(1,2), mean_vals_2014(1,2), mean_vals_2015(1,2), mean_vals_2016(1,2), mean_vals_2017(1,2), mean_vals_2018(1,2));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_vals_2010(1,1), std_vals_2011(1,2),...
    std_vals_2012(1,2), std_vals_2013(1,2), std_vals_2014(1,2), std_vals_2015(1,2), std_vals_2016(1,2), std_vals_2017(1,2), std_vals_2018(1,2));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { $h_0$}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_vals_2010(1,2), mean_vals_2011(1,2),...
    mean_vals_2012(1,2), mean_vals_2013(1,2), mean_vals_2014(1,2), mean_vals_2015(1,2), mean_vals_2016(1,2), mean_vals_2017(1,2), mean_vals_2018(1,2));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_vals_2010(1,1), std_vals_2011(1,2),...
    std_vals_2012(1,2), std_vals_2013(1,2), std_vals_2014(1,2), std_vals_2015(1,2), std_vals_2016(1,2), std_vals_2017(1,2), std_vals_2018(1,2));
fprintf(FID, '\\cmidrule(r){1-10} \\\\\n');
fprintf(FID, ' { $h_0$ est}& $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ \\\\\n', mean_vals_2010(1,2), mean_vals_2011(1,2),...
    mean_vals_2012(1,2), mean_vals_2013(1,2), mean_vals_2014(1,2), mean_vals_2015(1,2), mean_vals_2016(1,2), mean_vals_2017(1,2), mean_vals_2018(1,2));
fprintf(FID, ' & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$ & $(%.4f)$& $(%.4f)$& $(%.4f)$ \\\\\n', std_vals_2010(1,1), std_vals_2011(1,2),...
    std_vals_2012(1,2), std_vals_2013(1,2), std_vals_2014(1,2), std_vals_2015(1,2), std_vals_2016(1,2), std_vals_2017(1,2), std_vals_2018(1,2));
fprintf(FID, '\\bottomrule\n');
fprintf(FID, '\\end{tabular}}\n\\end{tabularx}}\n');

fprintf(FID, '\n  ');


fprintf(FID, '\\vspace{3 cm}\n');

fprintf(FID, '\n  ');



fprintf(FID, '\\end{document}');
fclose(FID);