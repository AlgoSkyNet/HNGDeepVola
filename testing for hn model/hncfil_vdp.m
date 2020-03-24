function price=hncfil_vdp(S0,K,n2,T)
theta=[2.231e+00 2.101e-17 3.3127e-06 9.012e-01 1.276e+02];
rf=.00001;
lambda=theta(1);
omega=theta(2);
a=theta(3);
b=theta(4);
c=theta(5);
R=11;
price=real(integral(@Integrand,R-1i*1000,R+1i*1000)*exp(-rf*T)/(2*pi*1i));
function g=Integrand(phi)
g=characterfun(phi);
end
function f=characterfun(phi)
phi=phi';    
lambdaq=lambda*(1-2*a*n2);               
cq=c*(1-2*a*n2);              
sigma=(omega+a)/(1-b-a*cq*cq);
sigmaq=sigma/(1-2*a*n2);
lq=lambdaq+cq+.5;
omegaq=omega/(1-2*a*n2);
aq=a/(1-2*a*n2)^2;
A(:,T-1)=phi.*rf;
B(:,T-1)=phi.*(lambdaq+cq)-.5*lq^2+.5*(phi-lq).^2;
for i=2:T-1
    A(:,T-i)=A(:,T-i+1)+phi.*rf+B(:,T-i+1).*omegaq-.5*log(1-2*aq*B(:,T-i+1));
    B(:,T-i)=phi.*(lambdaq+cq)-.5*lq^2+b*B(:,T-i+1)+.5*(phi-lq).^2./(1-2*aq*B(:,T-i+1));
end
At=A(:,1)+phi.*rf+B(:,1).*omegaq-.5*log(1-2*aq*B(:,1));                    
Bt=phi.*(lambdaq+cq)-.5*lq^2+b*B(:,1)+.5*(phi-lq).^2./(1-2.*aq.*B(:,1)); 
f=S0.^phi.*exp(At+Bt.*sigmaq+(1-phi)*log(K)-log(phi)-log(phi-1));
f=f' ;
end
end