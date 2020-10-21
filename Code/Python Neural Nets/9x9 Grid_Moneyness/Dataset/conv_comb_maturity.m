function [pos,share] =conv_comb_maturity(x)
    M   = 10:30:250;
    matu = floor((x-10)/30)*30+10;
    if x<10
        pos = 1;
        share = 1;
    else
        pos = find(M==matu);
        share  = (matu+30-x)/30;   
    end
end