function [mn_loglik, h, h_all] = ll_hng_n_paper(par0,x,r,h0)
n = length(x);
omega = par0(1);
alpha = par0(2);
beta = par0(3);
gamma = par0(4);
lambda = par0(5);
h = h0;
loglik = 0;
h = omega+alpha*(-(gamma)*h)^2+beta*h;
loglik = loglik-1/2*log(2*pi)-0.5*log(h)-0.5*(x(1)-r-lambda*h)^2/h;
h_all(1) = h;
for i = 2:n
    h = omega+alpha*(x(i)-r-(gamma+lambda)*h)^2/h+beta*h;
    h_all(i) = h;
    loglik = loglik-1/2*log(2*pi)-0.5*log(h)-0.5*(x(i)-r-lambda*h)^2/h;
end
mn_loglik = -loglik;