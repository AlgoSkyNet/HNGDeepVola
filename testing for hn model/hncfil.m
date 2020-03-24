function price=hncfil(S0,K,theta,T)
%rf=.00001;
rf                =   0.005/252;
%lambda=theta(1);
omega=theta(1);
a=theta(2);
b=theta(3);
cq=theta(4);
R=11;
price=real(integral(@Integrand,R-1i*1000,R+1i*1000)*exp(-rf*T)/(2*pi*1i));
function g=Integrand(phi)
g=characterfun(phi);
end
function f=characterfun(phi)
phi=phi';    
lambdaq=-.5;               
%cq=c+lambda+.5;              
%sigma=(omega+a)/(1-b-a*cq*cq);
A(:,T-1)=phi.*rf;
B(:,T-1)=phi.*(lambdaq+cq)-.5*cq^2+.5*(phi-cq).^2;
for i=2:T-1
    A(:,T-i)=A(:,T-i+1)+phi.*rf+B(:,T-i+1).*omega-.5*log(1-2*a*B(:,T-i+1));
    B(:,T-i)=phi.*(lambdaq+cq)-.5*cq^2+b*B(:,T-i+1)+.5*(phi-cq).^2./(1-2*a*B(:,T-i+1));
end
At=A(:,1)+phi.*rf+B(:,1).*omega-.5*log(1-2*a*B(:,1));                    
Bt=phi.*(lambdaq+cq)-.5*cq^2+b*B(:,1)+.5*(phi-cq).^2./(1-2.*a.*B(:,1)); 

f=S0.^phi.*exp(At+Bt.*sigma+(1-phi)*log(K)-log(phi)-log(phi-1));
f=f' ;
end
end