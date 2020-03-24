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
%if 2 integral formula is needed please go to hngarchcalibratevdp and
%there replace hncfil_vdp with hngarchoptioncfvdp 
[x,val,history]=hngarchcalibratevdp(29819,-Inf,Inf,data);
n2hat=x;
mse=zeros(nopt,1);
n2=zeros(nopt,1);
mse(1)=history.fval(end,:);
n2(1)=n2hat;
r_mse=zeros(nopt,1);
mop=zeros(nopt,1);
P=data;
nset=length(P);
for i=1:nset
     %replace hncfil_vdp with hngarchoptioncfvdp to use 2 integral formula
P(i,6)=(P(i,3)-hncfil_vdp(P(i,1),P(i,2),n2(1),P(i,4)))^2;
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
%if 2 integral formula is needed please go to hngarchcalibratevdp and
% there replace hncfil_vdp with hngarchoptioncfvdp
[x,val,history]=hngarchcalibratevdp(n2(i-1),-Inf,Inf,data);
n2hat=x;
mse(i)=history.fval(end,:);
n2(i)=n2hat;
P=data;
nset=length(P);
for j=1:nset
    %replace hncfil_vdp with hngarchoptioncfvdp to use 2 integral formula
P(j,6)=(P(j,3)-hncfil_vdp(P(j,1),P(j,2),n2(i),P(j,4)))^2;
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
