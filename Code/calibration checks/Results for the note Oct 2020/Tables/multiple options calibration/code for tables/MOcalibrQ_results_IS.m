clear;
ifHalfYear = 0;
if ifHalfYear
    periodstr = '_6m';
else
    periodstr = '_12m';
end

year_nums = {'2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'};
num_years = length(year_nums);

%load(strcat('data_MO_IS', periodstr, '_rAvYieldMLEP_rAvYieldCalibr_10_18.mat'));
load(strcat('data_MO_IS', periodstr, '_rWeekTbillMLEP_rWeekTbillCalibr_10_18.mat'));


%FID = fopen(strcat('MO_IS_res', periodstr, '_rAvYieldMLEP_rAvYieldCalibr_10_18.tex'), 'w');
FID = fopen(strcat('MO_IS_res', periodstr, '_rWeekTbillMLEP_rWeekTbillCalibr_10_18.tex'), 'w');

fprintf(FID, '%%&pdflatex \r%%&cont-en \r%%&pdftex \r');
fprintf(FID, '\\documentclass[10pt]{article} \n\\usepackage{latexsym,amsmath,amssymb,graphics,amscd} \n');
fprintf(FID, '\\usepackage{multirow} \n\\usepackage{booktabs} \n');
fprintf(FID, '\\usepackage{tabularx} \n\\usepackage[hang,footnotesize]{caption} \n');
fprintf(FID, '\\usepackage[pdftex]{graphicx} \n\\usepackage{color}\n\\textwidth15.8 cm\n\\textheight20.8 cm\n\\oddsidemargin.4cm\n\\evensidemargin.4cm \n\\begin{document} \n');

%fprintf(FID, '\\noindent\\begin{center} Results are obtained with r average yield over the calibration period, MLE P parameters obtained with average yield r over 10 years \\end{center} \n');
fprintf(FID, '\\noindent\\begin{center} Results are obtained with r weekly tbill over the calibration period, MLE P parameters obtained with weekly tbill r \\end{center} \n');

fprintf(FID, '\\noindent\\makebox[\\textwidth]{ \n');
fprintf(FID, '\\begin{tabularx}{1.3\\textwidth}{X} \n \\scalebox{0.7}{ \n\\begin{tabular}{ccccccccccc} \n');
fprintf(FID, '\\toprule \n');
if ifHalfYear
    fprintf(FID, '\\multicolumn{11}{c}{{\\bf MULTIPLE OPIIONS CALIBRATION EXERCISE OVER 6 MONTHS, IN-SAMPLE RESULTS}} \\\\\n');
else
    fprintf(FID, '\\multicolumn{11}{c}{{\\bf MULTIPLE OPIIONS CALIBRATION EXERCISE OVER 12 MONTHS, IN-SAMPLE RESULTS}} \\\\\n');
end

fprintf(FID, '\\midrule \n');
param_ind = 0;
for cur_num = 1:num_years 
if cur_num == 1
fprintf(FID, ' { {\\bf values}}&{ $\\omega$}& { $\\alpha$} & { $\\beta$} & { $\\gamma^{*}$} & { $h_0^Q$ } & { {\\bf persistency}} & { {\\bf OptLL} }& { {\\bf normOptLL} } & { {\\bf MSE} } & { {\\bf IVRMSE} } \\\\\n');
fprintf(FID, '\\cmidrule(r){1-11} \\\\\n');
end
fprintf(FID, strcat('\\multicolumn{11}{c}{{\\bf', year_nums{cur_num},  '}} \\\\\n'));
fprintf(FID, '\\cmidrule(r){1-11} \n');

param_ind = param_ind + 1;
fprintf(FID, ' { {\\bf h0 P}}& $%.4e$ & $%.4e$ & $%.4f$ & $%.4f$ & $%.4e$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ &$%.4f$\\\\\n', [row(param_ind,1:5),row(param_ind,3)+row(param_ind,2)*row(param_ind,4).^2, row(param_ind,6:9)]);

param_ind = param_ind + 1;
fprintf(FID, ' { {\\bf h0 RV}}& $%.4e$ & $%.4e$ & $%.4f$ & $%.4f$ & $%.4e$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ &$%.4f$\\\\\n', [row(param_ind,1:5),row(param_ind,3)+row(param_ind,2)*row(param_ind,4).^2, row(param_ind,6:9)]);

param_ind = param_ind + 1;
fprintf(FID, ' { {\\bf h0 Q}}& $%.4e$ & $%.4e$ & $%.4f$ & $%.4f$ & $%.4e$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ &$%.4f$\\\\\n', [row(param_ind,1:5),row(param_ind,3)+row(param_ind,2)*row(param_ind,4).^2, row(param_ind,6:9)]);

param_ind = param_ind + 1;
fprintf(FID, ' { {\\bf h0 est}}& $%.4e$ & $%.4e$ & $%.4f$ & $%.4f$ & $%.4e$ & $%.4f$ & $%.4f$& $%.4f$& $%.4f$ &$%.4f$\\\\\n', [row(param_ind,1:5),row(param_ind,3)+row(param_ind,2)*row(param_ind,4).^2, row(param_ind,6:9)]);


fprintf(FID, '\\bottomrule\n');
end
fprintf(FID, '\\end{tabular}}\n\\end{tabularx}}\n');

fprintf(FID, '\n  ');


fprintf(FID, '\\vspace{3 cm}\n');

fprintf(FID, '\n  ');



fprintf(FID, '\\end{document}');
fclose(FID);