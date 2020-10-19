clear

serverNames     = {'MSPablo', 'Sargas', 'local'};
useServer = 3;
switch useServer
    case 1
        pathPrefix = 'C:/Users/Lyudmila/Documents/GitHub/HenrikAlexJP/';
    case 2
        pathPrefix = 'C:/GIT/HenrikAlexJP/';
    otherwise
        pathPrefix = '/Users/lyudmila/Dropbox/GIT/HenrikAlexJP/';
end



ifHalfYear = 0;
useYield = 1;
useAverageWhenCal = 1;

for currentYear = 2010:2018
    for useScenario = 1:4
        if useScenario == 3
            fileNameWithMLEPests = strcat(pathPrefix, 'Code/calibration checks/Calibration MLE P/paper version/data for tables/Yields/r average/estimated h0P/Results with estimated h0P rAv/weekly_', num2str(currentYear), '_mle_opt_h0est_rAv_Unc');
        else
            fileNameWithMLEPests = strcat(pathPrefix, 'Code/calibration checks/Calibration MLE P/paper version/data for tables/Yields/r average/estimated h0P/Results with estimated h0P rAv/weekly_', num2str(currentYear), '_mle_opt_h0est_rAv');
        end
        calibrateMO(useServer, ifHalfYear, currentYear, useYield, useScenario, useAverageWhenCal, fileNameWithMLEPests);
    end
end
	