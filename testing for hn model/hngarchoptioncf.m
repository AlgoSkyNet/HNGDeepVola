function price=hngarchoptioncf(S0,K,theta,T)
rf                =   0.005/252;
%rf=.00001;
%lambda=theta(1);
omega=theta(2);
a=theta(3);
b=theta(4);
cq=theta(5);
price=.5*S0+(exp(-rf*T)/pi)*integral(@Integrand1,0,100)-K*exp(-rf*T)*(.5+(1/pi)*integral(@Integrand2,0,100));
function f1=Integrand1(phi)
f1=real((K.^(-1i*phi).*characterfun(1i*phi+1))./(1i*phi));
end
function f2=Integrand2(phi)
f2=real((K.^(-1i*phi).*characterfun(1i*phi))./(1i*phi));
end
function f=characterfun(phi)
phi=phi';    
lambdaq=-.5;               
%cq=c+lambda+.5;              
sigma=(omega+a)/(1-b-a*cq*cq);
A(:,T-1)=phi.*rf;
B(:,T-1)=phi.*(lambdaq+cq)-.5*cq^2+.5*(phi-cq).^2;
for i=2:T-1
    A(:,T-i)=A(:,T-i+1)+phi.*rf+B(:,T-i+1).*omega-.5*log(1-2*a*B(:,T-i+1));
    B(:,T-i)=phi.*(lambdaq+cq)-.5*cq^2+b*B(:,T-i+1)+.5*(phi-cq).^2./(1-2*a*B(:,T-i+1));
end
At=A(:,1)+phi.*rf+B(:,1).*omega-.5*log(1-2*a*B(:,1));                    
Bt=phi.*(lambdaq+cq)-.5*cq^2+b*B(:,1)+.5*(phi-cq).^2./(1-2.*a.*B(:,1)); 
f=S0.^phi.*exp(At+Bt.*sigma);
f=f' ;
end
end