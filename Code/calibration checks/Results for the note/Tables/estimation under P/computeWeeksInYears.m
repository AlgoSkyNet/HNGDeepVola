clear;
num_years = 9;
num_weeks = zeros(num_years, 1);
load('weekly_2010_mle_opt.mat');
i = 1;
num_weeks(i) = length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2010 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2010 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2011_mle_opt.mat')
i = i + 1;
num_weeks(i) = length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2011 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2011 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2012_mle_opt.mat')
i = i + 1;
num_weeks(i) = length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2012 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2012 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2013_mle_opt.mat')
i = i + 1;
num_weeks(i) = length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2013 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2013 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2014_mle_opt.mat')
i = i + 1;
num_weeks(i) = length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2014 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2014 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2015_mle_opt.mat')
i = i + 1;
num_weeks(i) = length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2015 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2015 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2016_mle_opt.mat')
i = i + 1;
num_weeks(i) = length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2016 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2016 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2017_mle_opt.mat')
i = i + 1;
num_weeks(i) = length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2017 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2017 = std(params_tmp(params_tmp(:,1)~=0, :));
load('weekly_2018_mle_opt.mat')
i = i + 1;
num_weeks(i) = length(params_tmp(params_tmp(:,1)~=0, :));
mean_vals_2018 = mean(params_tmp(params_tmp(:,1)~=0, :));
std_vals_2018 = std(params_tmp(params_tmp(:,1)~=0, :));