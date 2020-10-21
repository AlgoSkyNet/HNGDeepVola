function p = price_Q_order(params,data_vec,r)
% r is daily rate
%parloop efficiency variables
w = params(4);
a = params(1);
b = params(2);
g = params(3);
sig0  = params(5);
S =  data_vec(:,3);
K =  data_vec(:,1);
T =  data_vec(:,2);
pool_ = gcp();
pr(1:size(data_vec,1)) = parallel.FevalFuture;
%p=max(S-K,0)';%maximal intrinsic value
p = exp(-r.*T).*K; %upper bound call price
p = p';
for j =1:size(data_vec,1)
    pr(j) = parfeval(pool_,@HestonNandi_Q_oneintegral,1,S(j),K(j),sig0,T(j),r(j),w,a,b,g);
end
for j =1:size(data_vec,1)
    [completedIdx,value] = fetchNext(pr,0.5); %shutdown after 0.5s for integral calc
    p(completedIdx) = value;
    if p(completedIdx) < 0
        p(completedIdx) = 0;
    end
end
cancel(pr)
end