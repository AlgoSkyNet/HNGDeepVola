function [pos,share] =conv_comb_strike_moneyness(x)
    Moneyness   = 1.1:-0.025:0.9;
    strike = floor(40*x)/40;
    pos = find(round(Moneyness,3)==strike);
    share  = (strike+0.025-x)/0.025;   
end