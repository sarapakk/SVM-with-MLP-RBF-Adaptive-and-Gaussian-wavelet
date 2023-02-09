function k=kernel(u,v,p)
g=0;
for h=1:19
    g=(u(h,:)-v(h,:))^2+g;
end

k=exp(g/4)^p;