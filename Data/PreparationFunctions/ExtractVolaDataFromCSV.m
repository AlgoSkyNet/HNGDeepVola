%import from the csv file
clear;
T = readtable('oxfordmanrealizedvolatilityindices_090320.csv', 'Delimiter', ',');

[num_obs, num_cols] = size(T);

ind_names = unique(T(:,2));
[num_indeces, ~] = size(ind_names);
j = 1;
k = 1;
num_obs_year = 1;
ind_names = table2array(ind_names);
cur_var_name = matlab.lang.makeValidName(ind_names{j}, 'ReplacementStyle', 'delete');
for i = 1:num_obs  
    if strcmp(table2cell(T(i, 2)), ind_names{j})
        eval([cur_var_name '(k, :) = T(i, :);']);
        %eval([cur_year '=' cur_var_name '(k, 1);']);
        k = k + 1;
    else
        j = j + 1;
        k = 1;
        cur_var_name = matlab.lang.makeValidName(ind_names{j}, 'ReplacementStyle', 'delete');
    end
end
cur_year = 0;
num_years = 20;
num_obs_year = zeros(num_indeces, num_years);

for i = 27%1:num_indeces
    cur_var_name = matlab.lang.makeValidName(ind_names{i}, 'ReplacementStyle', 'delete');
    eval(['[cur_num_obs_ind, ~] = size(' cur_var_name ' );']);
    eval(['cur_year = datestr(datenum(table2cell(' cur_var_name '(1, 1)), ''yyyy-mm-dd HH:MM''), ''yyyy'');']);
    cur_year_num = str2num(cur_year);
    num_year = 1;
    num_obs_year(i, num_year) = 1;
    for j = 2:cur_num_obs_ind
        eval(['cur_year = datestr(datenum(table2cell(' cur_var_name '(j, 1)), ''yyyy-mm-dd HH:MM''), ''yyyy'');']);
        if cur_year_num == str2num(cur_year)
            num_obs_year(i, num_year) = num_obs_year(i, num_year) + 1;
        else
            cur_year_num = str2num(cur_year);
            num_year = num_year + 1;
            num_obs_year(i, num_year) = 1;
        end
    end
end
num_col_rv_to_annualize = 12;
cur_year = 0;
for i = 27%1:num_indeces
    cur_var_name = matlab.lang.makeValidName(ind_names{i}, 'ReplacementStyle', 'delete');
    eval(['[cur_num_obs_ind, ~] = size(' cur_var_name ' );']);

    k = 1;
    num_temp = num_obs_year(i, k);
    for j = 1:cur_num_obs_ind
        if j > num_temp
            k = k + 1;
            num_temp = num_temp + num_obs_year(i, k);
        end
        cur_year_ann = num_obs_year(i, k);
        eval([cur_var_name '(j, num_cols + 1) = table(sqrt(table2array(' cur_var_name '(j, num_col_rv_to_annualize))).*sqrt(cur_year_ann).*1e+2);']);

    end
end


vars = {'cur_var_name', 'i', 'ind_names',...
    'j','k','num_cols',...
    'num_indeces','num_obs', 'T',...
    'cur_num_obs_ind', 'cur_year', 'cur_year_ann', 'cur_year_num',...
    'num_col_rv_to_annualize', 'num_obs_year', 'num_temp', 'num_year', 'num_years', 'temp', 'ans'};

clear(vars{:});

save('SPX_volas_090320.mat');