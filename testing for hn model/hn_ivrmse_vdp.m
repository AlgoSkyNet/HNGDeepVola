clear
clc
tic
rf=.00001;
optionsdata = xlsread('calls2015.xlsx');
nod=unique(optionsdata(:,5));
nopt=length(nod);
filter=find(optionsdata(:,5)==nod(1)); 
for k=filter
    data=optionsdata(k,:);
end
%if 2 integral formula is needed please go to hngarchcalibratevdp and
%replace hncfil_vdp with hngarchoptioncfvdp
[x,val,history]=hngarchcalibratevdp(29819,-Inf,Inf,data);
n2hat=x;
mse=zeros(nopt,1);
n2=zeros(nopt,1);
mse(1)=history.fval(end,:);
n2(1)=n2hat;
ivrmse=zeros(nopt,1);
mop=zeros(nopt,1);
P=data;
nset=length(P);
for i=1:nset
P(i,6)=blsimpv(P(i,1),P(i,2),-1+(1+rf)^P(i,4),P(i,4)/252,P(i,3));
%replace hncfil with hngarchoptioncf to use 2 integral formula
P(i,7)=blsimpv(P(i,1),P(i,2),-1+(1+rf)^P(i,4),P(i,4)/252,hncfil_vdp(P(i,1),P(i,2),n2(1),P(i,4)));
if isnan(P(i,6)) || isnan(P(i,7))
    P(i,8)=0;
else
    P(i,8)=(100*(P(i,6)-P(i,7)))^2;
end
end
ivrmse(1)=sqrt(mean(P(:,8)));
mop(1)=mean(P(:,3));
for i=2:nopt
filter=find(optionsdata(:,5)==nod(i)); 
for k=filter
    data=optionsdata(k,:);
end
%if 2 integral formula is needed please go to hngarchcalibratevdp and
%replace hncfil_vdp with hngarchoptioncfvdp
[x,val,history]=hngarchcalibratevdp(n2(i-1),-Inf,Inf,data);
n2hat=x;
mse(i)=history.fval(end,:);
n2(i)=n2hat;
P=data;
nset=length(P);
for j=1:nset
P(j,6)=blsimpv(P(j,1),P(j,2),-1+(1+rf)^P(j,4),P(j,4)/252,P(j,3));
%replace hncfil with hngarchoptioncfvdp to use 2 integral formula
P(j,7)=blsimpv(P(j,1),P(j,2),-1+(1+rf)^P(j,4),P(j,4)/252,hncfil_vdp(P(j,1),P(j,2),n2(i),P(j,4)));
if isnan(P(j,6)) || isnan(P(j,7))
    P(j,8)=0;
else
    P(j,8)=(100*(P(j,6)-P(j,7)))^2;
end
end
ivrmse(i)=sqrt(mean(P(:,8)));
mop(i)=mean(P(:,3));
disp(i);
end
filename='hn_ivrmse.xlsx';
title={'n2','mse','mop','ivrmse'};
xlswrite(filename,title)
xlswrite(filename,[n2,mse,mop,ivrmse],1,'A2')
csvwrite('hn_ivrmse.csv',[n2,mse,mop,ivrmse])
T=table(n2,mse,mop,ivrmse,'VariableNames',{ 'n2', 'mse','mop','ivrmse'} );
writetable(T, 'hn_ivrmse.txt')
toc
