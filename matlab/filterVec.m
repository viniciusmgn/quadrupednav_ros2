function Y = filterVec(X)

N=5;
Y=[];
if size(X,1)==1
    X=X';
end


for i = 1: size(X,1)
i1 = max(i-N,1);
i2 = min(i+N,size(X,1));
Y = [Y; mean(X(i1:i2,:))];
end

end