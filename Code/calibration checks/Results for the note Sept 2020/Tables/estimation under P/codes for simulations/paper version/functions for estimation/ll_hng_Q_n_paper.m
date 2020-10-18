function [mn_loglik, sigma2_out] = ll_hng_Q_n_paper(par0,x,r,h0)
n = length(x);
omega = par0(1);
alpha = par0(2);
beta = par0(3);
gamma = par0(4);
h = h0;
loglik = 0;
sigma2_out = zeros(n, 1);
h = omega+alpha*((gamma-1/2))^2*h+beta*h;
sigma2_out(1) = h;
loglik = loglik-1/2*log(2*pi)-0.5*log(h)-0.5*(x(1)-r+1/2*h)^2/h;
for i = 2:n
    h = omega+alpha*(x(i-1)-r-(gamma-1/2)*h)^2/h+beta*h;
    sigma2_out(i) = h;
    loglik = loglik-1/2*log(2*pi)-0.5*log(h)-0.5*(x(i)-r+1/2*h)^2/h; 
end
mn_loglik = -loglik;