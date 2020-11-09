function [pos,share] =conv_comb_maturity_small(x)
    M   =  [9,15,22,30,50,80,110,140,170];
    if x<10
        pos = 1;
        share = 1;
    elseif x>169
        pos = 9;
        share = 1;
    else
        tmp = M-x;
        tmp(tmp>0) = -999;
        [~,pos] = max(tmp);
        share  = (M(pos+1)-x)/(M(pos+1)-M(pos));  
    end
end