function [x, fval, history] = hngarchcalibratevdp(x0,lb,ub,data)
    history.x = [];
    history.fval = [];
    options = optimoptions(@fmincon,'OutputFcn',@outfun,... 
    'Display','iter','Algorithm','interior-point','MaxIterations',1000);
    [x, fval] = fmincon(@hngarchmsevdp,x0,[],[],[],[],lb,ub,@constrainthngarch11vdp,options);
       
    function stop = outfun(x,optimValues,state)
        stop = false;
        if isequal(state,'iter')
             history.fval = [history.fval;optimValues.fval];
             history.x = [history.x;x];
        end
    end
    function mse=hngarchmsevdp(n2)
    P=data;
    n=length(P(:,1));
    diff=zeros(n,1);
    for i=1:n
        %replace hncfil_vdp with hngarchoptioncfvdp to use 2 integral formula
    diff(i)=P(i,3)-hncfil_vdp(P(i,1),P(i,2),n2,P(i,4));
    end
    mse=sqrt(mean(diff.^2));
    end
    function [con,coneq]=constrainthngarch11vdp(n2)
    theta=[2.231e+00 2.101e-17 3.3127e-06 9.012e-01 1.276e+02];    
    con=-1+2*theta(3)*n2;
    coneq=[];
    end
end    
    
    