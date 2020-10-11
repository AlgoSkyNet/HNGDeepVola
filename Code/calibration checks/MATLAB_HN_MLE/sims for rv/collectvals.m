j=1;
for i=2:length(values)
    optionsLikhngVal(j) = values{1,i}.optionsLikhng;
    params(j,:) = values{1,i}.hngparams;
%     optionsLikhngVal1(j) = values1{1,i}.optionsLikhng;
%     optionsLikhngVal2(j) = values2{1,i}.optionsLikhng;
%     params1(j,:) = values1{1,i}.hngparams;
%     params2(j,:) = values2{1,i}.hngparams;
    j = j+1;
end