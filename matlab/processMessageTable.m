function messages = processMessageTable(msgtable,generalCounter)

N = length(generalCounter);

messagesRaw = cell(N,1);

for i = 1: size(msgtable,1)
    ind = find(generalCounter==msgtable{i,1});
    for j = ind
        messagesRaw{j}{end+1} = msgtable{i,2};
    end
end

messages = @(i) dispmsg(messagesRaw,i);
end

function [] = dispmsg(msgRaw,i)

for j = 1: length(msgRaw{i})
    disp(msgRaw{i}{j});
end

end