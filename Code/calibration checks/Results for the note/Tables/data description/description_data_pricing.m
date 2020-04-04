clear;
load('description_data_pricing_calls_wednesdays_2010_2018.mat')

FID = fopen('description_data_pricing_calls_2010_2018.tex', 'w');
fprintf(FID, '%%&pdflatex \r%%&cont-en \r%%&pdftex \r');
fprintf(FID, '\\documentclass[10pt]{article} \n\\usepackage{latexsym,amsmath,amssymb,graphics,amscd} \n');
fprintf(FID, '\\usepackage{multirow} \n\\usepackage{booktabs} \n');
fprintf(FID, '\\usepackage{tabularx} \n\\usepackage[hang,footnotesize]{caption} \n');
fprintf(FID, '\\usepackage[pdftex]{graphicx} \n\\usepackage{color}\n\\textwidth15.8 cm\n\\textheight20.8 cm\n\\oddsidemargin.4cm\n\\evensidemargin.4cm \n\\begin{document} \n');

fprintf(FID, '\\noindent\\makebox[\\textwidth]{ \n');
fprintf(FID, '\\begin{tabularx}{1.3\\textwidth}{X} \n \\scalebox{0.85}{ \n\\begin{tabular}{llccccccc} \n');
fprintf(FID, '\\toprule \n');
fprintf(FID, '\\multicolumn{9}{c}{{\\bf BASIC FEATURES OF THE OPTION PRICING (PUTS AND CALLS) DATASET (WEDNESDAYS)}} \\\\\n');
fprintf(FID, '\\midrule \n');
fprintf(FID, '\\multicolumn{2}{c}{ }&\\multicolumn{6}{c}{{\\bf Moneyness $S _0/K $}}&{\\bf Across}\\\\ \n');
fprintf(FID, '\\cmidrule(r){3-8} \n');

fprintf(FID, ' &{\\bf Maturities}');
fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 0.9, 0.95);
fprintf(FID, '&\\multicolumn{1}{c}{$(%4.3f, %4.3f]$}', 0.95, 0.975);
fprintf(FID, '&\\multicolumn{1}{c}{$(%4.3f, %4.3f]$}', 0.975, 1);
fprintf(FID, '&\\multicolumn{1}{c}{$(%4.3f, %4.3f]$}', 1, 1.025);
fprintf(FID, '&\\multicolumn{1}{c}{$(%4.3f, %4.3f]$}', 1.025, 1.05);
fprintf(FID, '&\\multicolumn{1}{c}{$(%4.3f, %4.3f]$}', 1.05, 1.1);
fprintf(FID, '&{\\bf Moneyness}');

fprintf(FID, '\\\\\n');
fprintf(FID, '\\midrule \n');



fprintf(FID, '\\multirow{4}{*}{\\parbox{3.5cm}{\\centering {\\bf Number\\\\ of Contracts}}} \n');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq  T \\leq {\\bf %d}$} \n', 8, 30);
for ii=1:6
    fprintf(FID, '&%d',NumberOfContracts(1, ii));
end
fprintf(FID, '&%d',sum(NumberOfContracts(1, :)));
fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}< T \\leq {\\bf %d}$} \n', 30, 80);
for ii=1:6
    fprintf(FID, '&%d',NumberOfContracts(2, ii));
end
fprintf(FID, '&%d',sum(NumberOfContracts(2, :)));
fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}< T \\leq {\\bf %d}$} \n', 80, 180);
for ii=1:6
    fprintf(FID, '&%d',NumberOfContracts(3, ii));
end
fprintf(FID, '&%d',sum(NumberOfContracts(3, :)));
fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}< T \\leq {\\bf %d}$} \n', 180, 250);
for ii=1:6
    fprintf(FID, '&%d',NumberOfContracts(4, ii));
end
fprintf(FID, '&%d',sum(NumberOfContracts(4, :)));
fprintf(FID, '\\\\\n  ');

fprintf(FID, '\\midrule \n');
fprintf(FID, '\\midrule \n');


fprintf(FID, '{\\bf Across Maturities}&  ');
for ii=1:6
    fprintf(FID, '&%d',sum(NumberOfContracts(:, ii)));
end
fprintf(FID, '&%d',sum(NumberOfContracts(:)));

fprintf(FID, '\\\\\n  ');
fprintf(FID, '\\midrule \n');
fprintf(FID, '\\midrule \n');




fprintf(FID, '\\multirow{4}{*}{\\parbox{3.5cm}{\\centering {\\bf Average\\\\ Prices}}} \n');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq  T \\leq {\\bf %d}$} \n', 8, 30);
for ii=1:6
    if ~isnan(AveragePrices(1, ii))
        fprintf(FID, '&%4.3f',AveragePrices(1, ii));
    else
        fprintf(FID, '&------');
    end
end
fprintf(FID, '&%4.3f', nansum(AveragePrices(1, :).*NumberOfContracts(1, :))/sum(NumberOfContracts(1, :)));
fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}< T \\leq {\\bf %d}$} \n', 30, 80);
for ii=1:6
    if ~isnan(AveragePrices(2, ii))
        fprintf(FID, '&%4.3f',AveragePrices(2, ii));
    else
        fprintf(FID, '&------');
    end
end
fprintf(FID, '&%4.3f', nansum(AveragePrices(2, :).*NumberOfContracts(2, :))/sum(NumberOfContracts(2, :)));
fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}< T \\leq {\\bf %d}$} \n', 80, 180);
for ii=1:6
    if ~isnan(AveragePrices(3, ii))
        fprintf(FID, '&%4.3f',AveragePrices(3, ii));
    else
        fprintf(FID, '&------');
    end
end
fprintf(FID, '&%4.3f', nansum(AveragePrices(3, :).*NumberOfContracts(3, :))/sum(NumberOfContracts(3, :)));
fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}< T \\leq {\\bf %d}$} \n', 180, 250);
for ii=1:6
    if ~isnan(AveragePrices(4, ii))
        fprintf(FID, '&%4.3f',AveragePrices(4, ii));
    else
        fprintf(FID, '&------');
    end
end
fprintf(FID, '&%4.3f', nansum(AveragePrices(4, :).*NumberOfContracts(4, :))/sum(NumberOfContracts(4, :)));
fprintf(FID, '\\\\\n  ');

fprintf(FID, '\\midrule \n');
fprintf(FID, '\\midrule \n');

fprintf(FID, '{\\bf Across Maturities}& ');
for ii=1:6
    tempAveragePrices(ii) = nansum(AveragePrices(:, ii).*NumberOfContracts(:, ii));
    fprintf(FID, '&%4.3f',nansum(AveragePrices(:, ii).*NumberOfContracts(:, ii))/sum(NumberOfContracts(:, ii)));
end
fprintf(FID, '&%4.3f',nansum(tempAveragePrices(:))/sum(NumberOfContracts(:)));

fprintf(FID, '\\\\\n  ');
fprintf(FID, '\\midrule \n');
fprintf(FID, '\\midrule \n');

fprintf(FID, '\\multirow{4}{*}{\\parbox{3.5cm}{\\centering {\\bf Average\\\\ Implied Volatilities}}} \n');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq  T \\leq {\\bf %d}$} \n', 8, 30);
for ii=1:6
    if ~isnan(AverageImpliedVolatilities(1, ii))
        fprintf(FID, '&%4.3f',AverageImpliedVolatilities(1, ii));
    else
        fprintf(FID, '&------');
    end
end
fprintf(FID, '&%4.3f', nansum(AverageImpliedVolatilities(1, :).*NumberOfContracts(1, :))/sum(NumberOfContracts(1, :)));
fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}< T \\leq {\\bf %d}$} \n', 30, 80);
for ii=1:6
    if ~isnan(AverageImpliedVolatilities(2, ii))
        fprintf(FID, '&%4.3f',AverageImpliedVolatilities(2, ii));
    else
        fprintf(FID, '&------');
    end
end
fprintf(FID, '&%4.3f', nansum(AverageImpliedVolatilities(2, :).*NumberOfContracts(2, :))/sum(NumberOfContracts(2, :)));
fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}< T \\leq {\\bf %d}$} \n', 80, 180);
for ii=1:6
    if ~isnan(AverageImpliedVolatilities(3, ii))
        fprintf(FID, '&%4.3f',AverageImpliedVolatilities(3, ii));
    else
        fprintf(FID, '&------');
    end
end
fprintf(FID, '&%4.3f', nansum(AverageImpliedVolatilities(3, :).*NumberOfContracts(3, :))/sum(NumberOfContracts(3, :)));
fprintf(FID, '\\\\\n  ');
fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}< T \\leq {\\bf %d}$} \n', 180, 250);
for ii=1:6
    if ~isnan(AverageImpliedVolatilities(4, ii))
        fprintf(FID, '&%4.3f',AverageImpliedVolatilities(4, ii));
    else
        fprintf(FID, '&------');
    end
end
fprintf(FID, '&%4.3f', nansum(AverageImpliedVolatilities(4, :).*NumberOfContracts(4, :))/sum(NumberOfContracts(4, :)));
fprintf(FID, '\\\\\n  ');

fprintf(FID, '\\midrule \n');
fprintf(FID, '\\midrule \n');

fprintf(FID, '{\\bf Across Maturities}& ');
for ii=1:6
    tempAverageImpliedVolatilities(ii) = nansum(AverageImpliedVolatilities(:, ii).*NumberOfContracts(:, ii));
    fprintf(FID, '&%4.3f',nansum(AverageImpliedVolatilities(:, ii).*NumberOfContracts(:, ii))/sum(NumberOfContracts(:, ii)));
end
fprintf(FID, '&%4.3f',nansum(tempAverageImpliedVolatilities(:))/sum(NumberOfContracts(:)));

fprintf(FID, '\\\\\n  ');

fprintf(FID, '\\midrule \n');

fprintf(FID, '\\bottomrule\n');
fprintf(FID, '\\end{tabular}}\n\\end{tabularx}}\n');

fprintf(FID, '\n  ');


fprintf(FID, '\\vspace{3 cm}\n');

fprintf(FID, '\n  ');

% For thursdays

% load('description_data_pricing_thursdays_2007_2013.mat')
% 
% 
% fprintf(FID, '\\noindent\\makebox[\\textwidth]{ \n');
% fprintf(FID, '\\begin{tabularx}{1.3\\textwidth}{X} \n \\scalebox{0.85}{ \n\\begin{tabular}{llccccccc} \n');
% fprintf(FID, '\\toprule \n');
% fprintf(FID, '\\multicolumn{9}{c}{{\\bf BASIC FEATURES OF THE OPTION PRICING (PUTS AND CALLS) DATASET (THURSDAYS)}} \\\\\n');
% fprintf(FID, '\\midrule \n');
% fprintf(FID, '\\multicolumn{2}{c}{ }&\\multicolumn{6}{c}{{\\bf Moneyness $S _0/K $}}&{\\bf Across}\\\\ \n');
% fprintf(FID, '\\cmidrule(r){3-8} \n');
% 
% fprintf(FID, ' &{\\bf Maturities}');
% fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 0.9, 0.95);
% fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 0.95, 0.975);
% fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 0.975, 1);
% fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 1, 1.025);
% fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 1.025, 1.05);
% fprintf(FID, '&\\multicolumn{1}{c}{$[%4.3f, %4.3f]$}', 1.05, 1.1);
% fprintf(FID, '&{\\bf Moneyness}');
% 
% fprintf(FID, '\\\\\n');
% fprintf(FID, '\\midrule \n');
% 
% 
% 
% fprintf(FID, '\\multirow{4}{*}{\\parbox{3.5cm}{\\centering {\\bf Number\\\\ of Contracts}}} \n');
% fprintf(FID, '&\\multirow{1}{*}{\\makebox[2cm]{$  T < {\\bf %d}$}} \n', 30);
% for ii=1:6
%     fprintf(FID, '&%d',NumberOfContracts(1, ii));
% end
% fprintf(FID, '&%d',sum(NumberOfContracts(1, :)));
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T < {\\bf %d}$} \n', 30, 80);
% for ii=1:6
%     fprintf(FID, '&%d',NumberOfContracts(2, ii));
% end
% fprintf(FID, '&%d',sum(NumberOfContracts(2, :)));
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T < {\\bf %d}$} \n', 80, 180);
% for ii=1:6
%     fprintf(FID, '&%d',NumberOfContracts(3, ii));
% end
% fprintf(FID, '&%d',sum(NumberOfContracts(3, :)));
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T \\leq {\\bf %d}$} \n', 180, 250);
% for ii=1:6
%     fprintf(FID, '&%d',NumberOfContracts(4, ii));
% end
% fprintf(FID, '&%d',sum(NumberOfContracts(4, :)));
% fprintf(FID, '\\\\\n  ');
% 
% fprintf(FID, '\\midrule \n');
% fprintf(FID, '\\midrule \n');
% 
% 
% fprintf(FID, '{\\bf Across Maturities}&  ');
% for ii=1:6
%     fprintf(FID, '&%d',sum(NumberOfContracts(:, ii)));
% end
% fprintf(FID, '&%d',sum(NumberOfContracts(:)));
% 
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '\\midrule \n');
% fprintf(FID, '\\midrule \n');
% 
% 
% 
% 
% fprintf(FID, '\\multirow{4}{*}{\\parbox{3.5cm}{\\centering {\\bf Average\\\\ Prices}}} \n');
% fprintf(FID, '&\\multirow{1}{*}{\\makebox[2cm]{$  T < {\\bf %d}$}} \n', 30);
% for ii=1:6
%     if ~isnan(AveragePrices(1, ii))
%         fprintf(FID, '&%4.3f',AveragePrices(1, ii));
%     else
%         fprintf(FID, '&------');
%     end
% end
% fprintf(FID, '&%4.3f', nansum(AveragePrices(1, :).*NumberOfContracts(1, :))/sum(NumberOfContracts(1, :)));
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T < {\\bf %d}$} \n', 30, 80);
% for ii=1:6
%     if ~isnan(AveragePrices(2, ii))
%         fprintf(FID, '&%4.3f',AveragePrices(2, ii));
%     else
%         fprintf(FID, '&------');
%     end
% end
% fprintf(FID, '&%4.3f', nansum(AveragePrices(2, :).*NumberOfContracts(2, :))/sum(NumberOfContracts(2, :)));
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T < {\\bf %d}$} \n', 80, 180);
% for ii=1:6
%     if ~isnan(AveragePrices(3, ii))
%         fprintf(FID, '&%4.3f',AveragePrices(3, ii));
%     else
%         fprintf(FID, '&------');
%     end
% end
% fprintf(FID, '&%4.3f', nansum(AveragePrices(3, :).*NumberOfContracts(3, :))/sum(NumberOfContracts(3, :)));
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T \\leq {\\bf %d}$} \n', 180, 250);
% for ii=1:6
%     if ~isnan(AveragePrices(4, ii))
%         fprintf(FID, '&%4.3f',AveragePrices(4, ii));
%     else
%         fprintf(FID, '&------');
%     end
% end
% fprintf(FID, '&%4.3f', nansum(AveragePrices(4, :).*NumberOfContracts(4, :))/sum(NumberOfContracts(4, :)));
% fprintf(FID, '\\\\\n  ');
% 
% fprintf(FID, '\\midrule \n');
% fprintf(FID, '\\midrule \n');
% 
% fprintf(FID, '{\\bf Across Maturities}& ');
% for ii=1:6
%     tempAveragePrices(ii) = nansum(AveragePrices(:, ii).*NumberOfContracts(:, ii));
%     fprintf(FID, '&%4.3f',nansum(AveragePrices(:, ii).*NumberOfContracts(:, ii))/sum(NumberOfContracts(:, ii)));
% end
% fprintf(FID, '&%4.3f',nansum(tempAveragePrices(:))/sum(NumberOfContracts(:)));
% 
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '\\midrule \n');
% fprintf(FID, '\\midrule \n');
% 
% fprintf(FID, '\\multirow{4}{*}{\\parbox{3.5cm}{\\centering {\\bf Average\\\\ Implied Volatilities}}} \n');
% fprintf(FID, '&\\multirow{1}{*}{\\makebox[2cm]{$  T < {\\bf %d}$}} \n', 30);
% for ii=1:6
%     if ~isnan(AverageImpliedVolatilities(1, ii))
%         fprintf(FID, '&%4.3f',AverageImpliedVolatilities(1, ii));
%     else
%         fprintf(FID, '&------');
%     end
% end
% fprintf(FID, '&%4.3f', nansum(AverageImpliedVolatilities(1, :).*NumberOfContracts(1, :))/sum(NumberOfContracts(1, :)));
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T < {\\bf %d}$} \n', 30, 80);
% for ii=1:6
%     if ~isnan(AverageImpliedVolatilities(2, ii))
%         fprintf(FID, '&%4.3f',AverageImpliedVolatilities(2, ii));
%     else
%         fprintf(FID, '&------');
%     end
% end
% fprintf(FID, '&%4.3f', nansum(AverageImpliedVolatilities(2, :).*NumberOfContracts(2, :))/sum(NumberOfContracts(2, :)));
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T < {\\bf %d}$} \n', 80, 180);
% for ii=1:6
%     if ~isnan(AverageImpliedVolatilities(3, ii))
%         fprintf(FID, '&%4.3f',AverageImpliedVolatilities(3, ii));
%     else
%         fprintf(FID, '&------');
%     end
% end
% fprintf(FID, '&%4.3f', nansum(AverageImpliedVolatilities(3, :).*NumberOfContracts(3, :))/sum(NumberOfContracts(3, :)));
% fprintf(FID, '\\\\\n  ');
% fprintf(FID, '&\\multirow{1}{*}{${\\bf %d}\\leq T \\leq {\\bf %d}$} \n', 180, 250);
% for ii=1:6
%     if ~isnan(AverageImpliedVolatilities(4, ii))
%         fprintf(FID, '&%4.3f',AverageImpliedVolatilities(4, ii));
%     else
%         fprintf(FID, '&------');
%     end
% end
% fprintf(FID, '&%4.3f', nansum(AverageImpliedVolatilities(4, :).*NumberOfContracts(4, :))/sum(NumberOfContracts(4, :)));
% fprintf(FID, '\\\\\n  ');
% 
% fprintf(FID, '\\midrule \n');
% fprintf(FID, '\\midrule \n');
% 
% fprintf(FID, '{\\bf Across Maturities}& ');
% for ii=1:6
%     tempAverageImpliedVolatilities(ii) = nansum(AverageImpliedVolatilities(:, ii).*NumberOfContracts(:, ii));
%     fprintf(FID, '&%4.3f',nansum(AverageImpliedVolatilities(:, ii).*NumberOfContracts(:, ii))/sum(NumberOfContracts(:, ii)));
% end
% fprintf(FID, '&%4.3f',nansum(tempAverageImpliedVolatilities(:))/sum(NumberOfContracts(:)));
% 
% fprintf(FID, '\\\\\n  ');
% 
% fprintf(FID, '\\midrule \n');
% 
% fprintf(FID, '\\bottomrule\n');
% fprintf(FID, '\\end{tabular}}\n\\end{tabularx}}\n');
% 
% fprintf(FID, '\n  ');
% 
% 
% fprintf(FID, '\\vspace{3 cm}\n');
% 
% fprintf(FID, '\n  ');

fprintf(FID, '\\end{document}');
fclose(FID);