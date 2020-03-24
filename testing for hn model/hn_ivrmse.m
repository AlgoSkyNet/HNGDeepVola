clear;
%rf=.00001;
rf = 0.005/252;
optionsdata = load('calls2015_LG.csv');
nod=unique(optionsdata(:,5));
nopt=length(nod);
filter=find(optionsdata(:,5)==nod(1)); 
for k=filter
    data=optionsdata(k,:);
end
ctime=zeros(nopt,1);
rtime=zeros(nopt,1);
tic
%if 2 integral formula is needed please go to hngarchcalibrate and
%replace hncfil with hngarchoptioncf
[x,val,history]=hngarchcalibrate([0.0000027738 0.000016969 0.23378 181.77],...
    [1e-17 1e-06 1e-01 1e+02],...
    [.01 .1 1 1000],data);
%[x,val,history]=hngarchcalibrate([5.6972 0.0000027738 0.000016969 0.23378 181.77],[1e+00 1e-17 1e-06 1e-01 1e+02],[10 .01 .1 1 1000],data);
ctime(1)=toc;
thetahat=x;
lambda=zeros(nopt,1);
omega=zeros(nopt,1);
a=zeros(nopt,1);
b=zeros(nopt,1);
c_rn=zeros(nopt,1);
ivrmse=zeros(nopt,1);
mop=zeros(nopt,1);
mse=zeros(nopt,1);
mse(1)=sqrt(history.fval(end,:));
c_rn(1)=thetahat(5);
omega(1)=thetahat(2);
a(1)=thetahat(3);
b(1)=thetahat(4);
lambda(1)=thetahat(1);
theta=[lambda(1) omega(1) a(1) b(1) c(1)];
nset=length(data);
tic
for i=1:nset
data(i,6)=blsimpv(data(i,1),data(i,2),-1+(1+rf)^data(i,4),data(i,4)/252,data(i,3));
%replace hncfil with hngarchoptioncf to use 2 integral formula
if hngarchoptioncf(data(i,1),data(i,2),theta,data(i,4))<0
    data(i,7) = NaN;
else
    
    data(i,7)=blsimpv(data(i,1),data(i,2),-1+(1+rf)^data(i,4),data(i,4)/252,hngarchoptioncf(data(i,1),data(i,2),theta,data(i,4)));
end

if isnan(data(i,6)) || isnan(data(i,7)) || hngarchoptioncf(data(i,1),data(i,2),theta,data(i,4))<0
    data(i,8)=NaN;
else
    data(i,8)=(100*(data(i,6)-data(i,7)))^2;
end
    data(i,9)=hngarchoptioncf(data(i,1),data(i,2),theta,data(i,4));
end
rtime(1)=toc;
ivrmse(1)=sqrt(mean(data(:,8)));
mop(1)=mean(data(:,3));
[fid,~] = fopen('hn_stepresults.txt','wt');
fprintf(fid, 'week %d mop %d rmse %d ivrmse %d lambda %d omega %d a %d b %d c %d ctime %d rtime %d\n\t\n', ...
1, mop(1),mse(1),ivrmse(1),lambda(1), omega(1), a(1), b(1), c(1),ctime(1),rtime(1));
for i=2:nopt
filter=find(optionsdata(:,5)==nod(i)); 
for k=filter
    data=optionsdata(k,:);
end
tic
%if 2 integral formula is needed please go to hngarchcalibrate and
%replace hncfil with hngarchoptioncf
[x,val,history]=hngarchcalibrate([lambda(i-1) omega(i-1) a(i-1) b(i-1) c(i-1)],[1e+00 1e-17 1e-06 1e-01 1e+02],[10 .01 .1 1 1000],data);
ctime(i)=toc;
thetahat=x;
mse(i)=sqrt(history.fval(end,:));
omega(i)=thetahat(2);
a(i)=thetahat(3);
b(i)=thetahat(4);
c(i)=thetahat(5);
lambda(i)=thetahat(1);
nset=length(data);
theta=[lambda(i) omega(i) a(i) b(i) c(i)];
tic
for j=1:nset
data(j,6)=blsimpv(data(j,1),data(j,2),-1+(1+rf)^data(j,4),data(j,4)/252,data(j,3));
%replace hncfil with hngarchoptioncf to use 2 integral formula
if hngarchoptioncf(data(j,1),data(j,2),theta,data(j,4))<0
    data(j,7) = data(j,6);
else
    
    data(j,7)=blsimpv(data(j,1),data(j,2),-1+(1+rf)^data(j,4),data(j,4)/252,hngarchoptioncf(data(j,1),data(j,2),theta,data(j,4)));
end
if isnan(data(j,6)) || isnan(data(j,7))
    data(j,8)=0;
else
    data(j,8)=(100*(data(j,6)-data(j,7)))^2;
end
end
rtime(i)=toc;
ivrmse(i)=sqrt(mean(data(:,8)));
mop(i)=mean(data(:,3));
fprintf(fid, 'week %d mop %d rmse %d ivrmse %d lambda %d omega %d a %d b %d c %d ctime %d rtime %d\n\t\n', ...
i, mop(i),mse(i),ivrmse(i),lambda(i), omega(i), a(i), b(i), c(i),ctime(i),rtime(i));
disp(i);
end
fclose(fid);
csvwrite('hn_ivrmse.csv',[lambda,omega,a,b,c,mse,mop,ivrmse])
T=table(lambda,omega,a,b,c,mse,mop,ivrmse,'VariableNames',{ 'lambda','omega','a','b','c', 'mse','mop','ivrmse'} );
writetable(T, 'hn_ivrmse.txt')

