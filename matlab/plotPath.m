function [] = plotPath(posPath,oriPath,R,H,color,skip)

hold on;
for k = 1: size(posPath,1)
    if mod(k,skip)==0
        rot = [cos(oriPath(k)) -sin(oriPath(k)); sin(oriPath(k)) cos(oriPath(k))];
        P = rot*[H/2 H/2 -H/2 -H/2;-R R R -R]+[posPath(k,1);posPath(k,2)];
        plot([P(1,1) P(1,2)],[P(2,1) P(2,2)],'--','linewidth',1,'Color',color);
        plot([P(1,2) P(1,3)],[P(2,2) P(2,3)],'--','linewidth',1,'Color',color);
        plot([P(1,3) P(1,4)],[P(2,3) P(2,4)],'--','linewidth',1,'Color',color);
        plot([P(1,4) P(1,1)],[P(2,4) P(2,1)],'--','linewidth',1,'Color',color);
    end
end

end