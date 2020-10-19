function [sigma2_out] = sim_hng_Q_n(par0,x,r,h0)
n = length(x);
omega = par0(1);
alpha = par0(2);
beta = par0(3);
gamma = par0(4);
h = h0;
%loglik = 0;
sigma2_out = zeros(n, 1);
for i = 1:n
    %loglik = loglik-1/2*log(2*pi)-0.5*log(h)-0.5*(x(i)-r+1/2*h)^2/h;
    h = omega+alpha*(x(i)-r-(gamma-1/2)*h)^2/h+beta*h;
    sigma2_out(i) = h;
end
