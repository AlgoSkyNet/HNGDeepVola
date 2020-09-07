function [pos,share] =conv_comb_strike(x)
    K   = 0.9:0.025:1.1;
    strike = floor(40*x)/40;
    pos = find(round(K,3)==strike);
    share  = (strike+0.025-x)/0.025;   
end