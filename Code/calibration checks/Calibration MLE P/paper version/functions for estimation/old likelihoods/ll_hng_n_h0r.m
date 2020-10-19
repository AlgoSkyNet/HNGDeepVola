function [mn_loglik,h] = ll_hng_n_h0r(par0,x)
n = length(x);
omega = par0(1);
alpha = par0(2);
beta = par0(3);
gamma = par0(4);
lambda = par0(5);
h = par0(6);
r = par0(7);

loglik = 0;
for i = 1:n
    loglik = loglik-1/2*log(2*pi)-0.5*log(h)-0.5*(x(i)-r-lambda*h)^2/h;
    h = omega+alpha*(x(i)-r-(gamma+lambda)*h)^2/h+beta*h;
end
mn_loglik = -loglik;