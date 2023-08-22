if ~exist('timeStamp')
    fileloader;

    %Compute some parameters to show
    minp = min(pointsKDTree{end});
    maxp = max(pointsKDTree{end});
    delta = max(maxp(1)-minp(1),maxp(2)-minp(2));
    meanp = (maxp+minp)/2;
    
    limits = [meanp(1)-0.55*delta meanp(1)+0.55*delta meanp(2)-0.55*delta meanp(2)+0.55*delta];
    N = size(timeStamp,2);

    timeStamp = timeStamp-timeStamp(1);
    dt = mean(diff(timeStamp));

    for j = 1: size(position,1)-1
        dt = (timeStamp(j+1)-timeStamp(j));
        realLinVelocity(j,:) = (position(j+1,:)-position(j,:))/dt;
        realAngVelocity(j) = (orientation(j+1)-orientation(j))/dt;
    end

    realLinVelocity = filterVec(realLinVelocity);
    realAngVelocity = filterVec(realAngVelocity);
end


close all;
%limits=[-10 10];



figure;
for i = 359:400
if mod(i,1)==0
    
    plot(0,0);
    hold on;
    
    
    plot(pointsKDTree{i}(:,1),pointsKDTree{i}(:,2),'bo');
    %plot(currentLidarPoints{i}(:,1),currentLidarPoints{i}(:,2),'go');
    plot(witnessDistance(i,1),witnessDistance(i,2),'r.','markersize',25);
    plot(currentGoalPosition(i,1), currentGoalPosition(i,2), 'm*', 'markersize', 20);
    plot(pointsFrontier{i}(:,1),pointsFrontier{i}(:,2),'o','Color',[0.5 0.5 0.5]);

    % if currentOmega(i)==0
    %     plotPath(plannedPos0{i},plannedOri0{i},param_boundingRadius,param_boundingHeight,'c',1);
    % end
    % if currentOmega(i)==2
    %     plotPath(plannedPosPZ{i},plannedOriPZ{i},param_boundingRadius,param_boundingHeight,'c',1);
    %     plotPath(plannedPosNZ{i},plannedOriNZ{i},param_boundingRadius,param_boundingHeight,'r',1);
    % end
    % if currentOmega(i)==-2
    %      plotPath(plannedPosNZ{i},plannedOriNZ{i},param_boundingRadius,param_boundingHeight,'c',1);
    %      plotPath(plannedPosPZ{i},plannedOriPZ{i},param_boundingRadius,param_boundingHeight,'r',1);
    % end
    plotPath(commitedPos{i},commitedOri{i},param_boundingRadius,param_boundingHeight,[1.00 0.49 0.31],3);
    %plot(commitedPos{i}(1,1),commitedPos{i}(1,2),'o','Color',[1.00 0.49 0.31]);
    %plot(commitedPos{i}(:,1), commitedPos{i}(:,2),'Color',[1.00 0.49 0.31]);

    %Draw graph
    for j = 1: size(graphNodes{i},1)
        plot(graphNodes{i}(j,1),graphNodes{i}(j,2),'k.','markersize',20);
    end
    for j = 1: size(graphEdges{i},1)
        inPosition = graphNodes{i}(graphEdges{i}(j,1)+1,:);
        outPosition = graphNodes{i}(graphEdges{i}(j,2)+1,:);
        plot([inPosition(1) outPosition(1)],[inPosition(2) outPosition(2)],'k-');
    end

    %Draw graph path, when applicable
    if planningState(i)==1 
        for j = 1: length(currentPath{i})
            ind = currentPath{i}(j)+1;
            plot(graphNodes{i}(ind,1), graphNodes{i}(ind,2), '.', 'markersize',20,'Color',[255,127,80]/255);
            text(graphNodes{i}(ind,1), graphNodes{i}(ind,2)+0.2,num2str(j),'Color',[255,127,80]/255);
        end
        plot(explorationPosition(i,1),explorationPosition(i,2),'.','markersize',30,'Color',[0.5 0.5 0.5]);
    end
    if planningState(i)==2
        plot(explorationPosition(i,1),explorationPosition(i,2),'.','markersize',30,'Color',[0.5 0.5 0.5]);
    end


    distAtual = round(100*distance(i));
    minDistAtual = round(100*min(distance(10:i)));

    if planningState(i)==0
        planState = 'goingToGlobal';
    end
    if planningState(i)==1
        planState = 'pathToExploration';
    end
    if planningState(i)==2
        planState = 'goingToExplore';
    end
    if planningState(i)==3
        planState = 'planning';
    end

    if currentOmega(i)==0
        omegaText = '0';
    end
    if currentOmega(i)==2
        omegaText = '+Z';
    end
    if currentOmega(i)==-2
        omegaText = '-Z';
    end

    text(limits(1)+1,limits(4)-1,['i = '  num2str(i) ', D = ' num2str(distAtual) ' cm, minD = ' num2str(minDistAtual) ' cm, state = ' planState ', rot = ' omegaText] , 'FontSize', 15);
    drawRobot(position(i,:),orientation(i),param_boundingRadius,param_boundingHeight);
    plot([position(i,1) position(i,1)+3*desLinVelocity(i,1)], [position(i,2) position(i,2)+3*desLinVelocity(i,2)], 'linewidth',2,'Color',[0.5 0  0.5]);
    
    axis([limits limits]);
    set(gcf,'Position',[82 23  1200 1200])
    drawnow;
    hold off;
    
end 
end


%Plot results

figure;
for k = 1:2
    subplot(1,3,k);
    plot(timeStamp(1:end-1), realLinVelocity(:,k),'b');
    hold on;
    plot(timeStamp, desLinVelocity(:,k),'r-','linewidth',1);
    hold off;
    xlabel('time (seg)');
    ylabel('velocity (m/s)');
end

subplot(1,3,3);
plot(timeStamp(1:end-1), realAngVelocity,'b');
hold on;
plot(timeStamp, desAngVelocity,'r-','linewidth',1);
hold off;
xlabel('time (seg)');
ylabel('velocity (rad/s)');


