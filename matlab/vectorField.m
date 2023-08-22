function v = vectorField(pos,ori,posPath,oriPath)



for i = 1: length(oriPath)
    D(i) = sqrt(0.5*norm(pos-posPath(i,:))^2+(1-cos(ori-oriPath(i))));
end

[dmin,ind] = min(D);

posStar = posPath(ind,:);
oriStar = oriPath(ind);

plot3(posPath(:,1),posPath(:,2),oriPath,'b'); 
hold on;
plot3(pos(1),pos(2),ori,'r.','markersize',30);
plot3(posStar(1),posStar(2),oriStar,'g.','markersize',30);
hold off;


N = [posStar(1)-pos(1); posStar(2)-pos(2); sin(oriStar-ori)];
N = N/(norm(N)+0.000001);

if ind==1
    dcos = cos(oriPath(2))-cos(oriPath(1));
    dsin = sin(oriPath(2))-sin(oriPath(1));
    dtheta = -sin(oriPath(1)) * dcos + cos(oriPath(1)) * dsin;
    T = [posPath(2,1)-posPath(1,1); posPath(2,2)-posPath(1,2); dtheta];
else
    dcos = cos(oriPath(ind))-cos(oriPath(ind-1));
    dsin = sin(oriPath(ind))-sin(oriPath(ind-1));
    dtheta = -sin(oriPath(ind)) * dcos + cos(oriPath(ind)) * dsin;
    %dtheta = oriPath(ind)-oriPath(ind-1);
    T = [posPath(ind+1,1)-posPath(ind,1); posPath(ind+1,2)-posPath(ind,2); dtheta];
end

T = T/(norm(T)+0.0001);

G = (2 / 3.14) * atan(2 * sqrt(dmin));
H = sqrt(1 - (1 - 0.000001) * G * G);

disp(T'*N);

v = 0.3 * (0.5 * G * N + H * T);



end