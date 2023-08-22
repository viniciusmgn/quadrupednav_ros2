function [] = drawRobot(p,theta,R,H)

rot = [cos(theta) -sin(theta); sin(theta) cos(theta)];
P = rot*[H/2-0.3 H/2 H/2-0.3 -H/2 -H/2;-R 0 R R -R]+[p(1);p(2)];

pgon = polyshape(P(1,:),P(2,:));
plot(pgon,'FaceColor',[238,232,170]/255,'FaceAlpha',1);

% plot([P(1,1) P(1,2)],[P(2,1) P(2,2)],'g','linewidth',2);
% plot([P(1,2) P(1,3)],[P(2,2) P(2,3)],'g','linewidth',2);
% plot([P(1,3) P(1,4)],[P(2,3) P(2,4)],'g','linewidth',2);
% plot([P(1,4) P(1,1)],[P(2,4) P(2,1)],'g','linewidth',2);


end