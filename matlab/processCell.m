function S = processCell(s)

if ~isempty(s)
    ind = find(isnan(s(:,1)));
    S={};

    ind = [0; ind];

    for j = 1: length(ind)-1
        S{end+1} = s(ind(j)+1:ind(j+1)-1,:);
    end

else
    S={};
end

end