function [c,ceq] = nonlincon_order(x,scale)
c = (x(1)*scale(1)).*((x(3)*(scale(3))).^2)+x(2)*scale(2)-1+1e-8;
ceq = [];