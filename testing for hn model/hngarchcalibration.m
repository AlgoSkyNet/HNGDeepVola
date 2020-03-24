clear
clc
tic
optionsdata = xlsread('calls2015.xlsx');
nod=unique(optionsdata(:,5));
nopt=length(nod);
filter=find(optionsdata(:,5)==nod(1)); 
for k=filter
    data=optionsdata(k,:);
end
%if 2 integral formula is needed please go to hngarchcalibrate and
%there replace hncfil with hngarchoptioncf
[x,val,history]=hngarchcalibrate([2.231e+00 2.101e-17 3.3127e-06 9.012e-01 1.276e+02],[1e+00 1e-17 1e-06 1e-01 1e+02],[10 .01 .1 1 1000],data);
thetahat=x;
lambda=zeros(nopt,1);
omega=zeros(nopt,1);
a=zeros(nopt,1);
b=zeros(nopt,1);
c=zeros(nopt,1);
r_mse=zeros(nopt,1);
mop=zeros(nopt,1);
mse=zeros(nopt,1);
mse(1)=history.fval(end,:);
c(1)=thetahat(5);
omega(1)=thetahat(2);
a(1)=thetahat(3);
b(1)=thetahat(4);
lambda(1)=thetahat(1);
theta=[lambda(1) omega(1) a(1) b(1) c(1)];
P=data;
nset=length(P);
for i=1:nset
    %replace hncfil with hngarchoptioncf to use 2 integral formula
P(i,6)=(P(i,3)-hncfil(P(i,1),P(i,2),theta,P(i,4)))^2;
end
P=sortrows(P,6);
se=P(:,6);
bottom=ceil(length(se)*.90);
P(bottom:length(se),:)=[];
r_mse(1)=sqrt(mean(P(:,6)));
mop(1)=mean(P(:,3));
for i=2:nopt
filter=find(optionsdata(:,5)==nod(i)); 
for k=filter
    data=optionsdata(k,:);
end
%if 2 integral formula is needed please go to hngarchcalibrate and
%there replace hncfil with hngarchoptioncf
[x,val,history]=hngarchcalibrate([lambda(i-1) omega(i-1) a(i-1) b(i-1) c(i-1)],[1e+00 1e-17 1e-06 1e-01 1e+02],[10 .01 .1 1 1000],data);
thetahat=x;
mse(i)=history.fval(end,:);
omega(i)=thetahat(2);
a(i)=thetahat(3);
b(i)=thetahat(4);
c(i)=thetahat(5);
lambda(i)=thetahat(1);
P=data;
nset=length(P);
theta=[lambda(i) omega(i) a(i) b(i) c(i)];
for j=1:nset
    %replace hncfil with hngarchoptioncf to use 2 integral formula
P(j,6)=(P(j,3)-hncfil(P(j,1),P(j,2),theta,P(j,4)))^2;
end
P=sortrows(P,6);
se=P(:,6);
bottom=ceil(length(se)*.90);
P(bottom:length(se),:)=[];
r_mse(i)=sqrt(mean(P(:,6)));
mop(i)=mean(P(:,3));
disp(i);
end
toc
