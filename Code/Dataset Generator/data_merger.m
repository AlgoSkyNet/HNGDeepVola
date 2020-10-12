%% small price cutter
function [priceset,volaset,vegaset] = data_merger(name1,name2,name3,threshold)
data   = load(name1(1));
price  = data.data_price;
data   = load(name2(1));
vola   = data.data_vola;
data   = load(name3(1));
vega   = data.data_vega;
n      = length(name1);
for i=2:n
    data       = load(name1(i));
    price_tmp  = data.data_price;
    data       = load(name2(i));
    vola_tmp   = data.data_vola;
    data       = load(name3(i));
    vega_tmp   = data.data_vega;
    price      = [price;price_tmp];
    vola       = [vola;vola_tmp];
    vega       = [vega;vega_tmp];
end
if threshold>0
    idx      = ~any(price(:,15:end)'<threshold);
    priceset = price(idx,:);
    volaset  = vola(idx,:);
    vegaset  = vega(idx,:);
else
    priceset = price;
    volaset  = vola;
    vegaset  = vega;
end
end