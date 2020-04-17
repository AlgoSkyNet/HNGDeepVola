%figures for thesis
load("dataset.mat")
figure
subplot(2,3,1)
histogram(xx(:,1),'Normalization','probability')
subplot(2,3,2)
histogram(xx(:,2),'Normalization','probability')
subplot(2,3,3)
histogram(xx(:,3),'Normalization','probability')
subplot(2,3,4)
histogram(xx(:,4),'Normalization','probability')
subplot(2,3,5)
histogram(xx(:,5),'Normalization','probability')
min_ = min(yy);
max_ = max(yy);
figure
y = 0.9:0.025:1.1;
x = 30:30:210;
[X,Y] = meshgrid(x,y);
surf(x,y,reshape(min_,9,7));hold on;
surf(x,y,reshape(max_,9,7));hold on;
set(gca,'Zscale','log')

