function [x, fval, history] = hngarchcalibrate(x0,lb,ub,data)
    history.x = [];
    history.fval = [];
    options = optimoptions(@fmincon,'OutputFcn',@outfun,... 
    'Display','iter','Algorithm','interior-point','MaxIterations',1000,'MaxFunctionEvaluations',1500,'TolFun',1e-6,'TolX',1e-6);
    [x, fval] = fmincon(@hngarchmse,x0,[],[],[],[],lb,ub,@constrainthngarch11,options);
       
    function stop = outfun(x,optimValues,state)
        stop = false;
        if isequal(state,'iter')
             history.fval = [history.fval;optimValues.fval];
             history.x = [history.x;x];
        end
    end
    function mse=hngarchmse(theta)
    n=length(data(:,1));
    diff=zeros(n,1);
    for i=1:n
        %replace hncfil with hngarchoptioncf to use 2 integral formula
    diff(i)=data(i,3)-hncfil(data(i,1),data(i,2),theta,data(i,4));
    %diff(i)=data(i,3)-hngarchoptioncf(data(i,1),data(i,2),theta,data(i,4));
    end
    mse=mean(diff.^2);
    end
    function [con,coneq]=constrainthngarch11(theta)
    con=-1+theta(3)+theta(2)*theta(4)^2+1e-6;
    coneq=[];
    end
end    
    
    