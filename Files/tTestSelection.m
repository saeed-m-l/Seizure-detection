% This have a little Problem
%{
function [train_S,test_S] = tTestSelection(trainData,testData,alpha)
% Perform t-test for each feature
p_values = zeros(1, size(trainData, 2));

for i = 1:size(trainData, 2)
    [~, p_values(i)] = ttest2(trainData(:, i), testData(:, i), 'Alpha', alpha);
end
% Select features based on p-values
selected_features = find(p_values < alpha);

% Extract selected features from both training and test datasets
train_S = trainData(:, selected_features);
test_S = testData(:, selected_features);
end
%}
function [train,test] = tTestSelection(train_data,test_data, train_labels, significance_level)
    num_features = size(train_data, 2);
    p_values = zeros(1, num_features);

    for i = 1:num_features
        [~, p_values(i)] = ttest2(train_data(train_labels == -1, i), train_data(train_labels == 1, i));
    end

    selected_features = find(p_values < significance_level);
    train = train_data(:,selected_features);
    test = test_data(:,selected_features);
end
