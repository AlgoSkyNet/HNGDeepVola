j=1;
for i=1:length(values)
    if ~isempty(values{1,i})
    optionsLikhngVal(j) = values{1,i}.optionsLikhng;
    params(j,:) = values{1,i}.hngparams;
   numprice(j,:) = length(values{1,i}.Price);
   minprice(j,:) = min(values{1,i}.Price);
   numpricefiltered(j,:) = sum(values{1,i}.Price>3/8);
%     optionsLikhngVal1(j) = values1{1,i}.optionsLikhng;
%     optionsLikhngVal2(j) = values2{1,i}.optionsLikhng;
%     params1(j,:) = values1{1,i}.hngparams;
%     params2(j,:) = values2{1,i}.hngparams;
    j = j+1;
    end
end