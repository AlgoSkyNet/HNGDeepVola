function price=hngarchoptioncfvdp(S0,K,n2,T)
theta=[2.231e+00 2.101e-17 3.3127e-06 9.012e-01 1.276e+02];
rf=.00001;
lambda=theta(1);
omega=theta(2);
a=theta(3);
b=theta(4);
c=theta(5);
price=.5*S0+(exp(-rf*T)/pi)*integral(@Integrand1,0,100)-K*exp(-rf*T)*(.5+(1/pi)*integral(@Integrand2,0,100));
function f1=Integrand1(phi)
f1=real((K.^(-1i*phi).*characterfun(1i*phi+1))./(1i*phi));
end
function f2=Integrand2(phi)
f2=real((K.^(-1i*phi).*characterfun(1i*phi))./(1i*phi));
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
f=S0.^phi.*exp(At+Bt.*sigmaq);
f=f' ;
end
end