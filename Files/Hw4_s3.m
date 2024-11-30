%% Part 1 ( ready file -- preapare data) - subject 1
Fs = 256;
% For sub 3
clear;close all;clc;
c1 = edfread("Sub3\WS3\chb03_02.edf");
c2 = edfread("Sub3\WS3\chb03_04.edf");
c3 = edfread("Sub3\WS3\chb03_34.edf");
c4 = edfread("Sub3\WS3\chb03_35.edf");
c5 = edfread("Sub3\WS3\chb03_36.edf");
c = edfread("Sub3\WOS3\chb03_17.edf");
s_time = [730,2161,1981,2591,1724];
s1 = getDataBeforeTime(c1, s_time(1));
s1 = getMatrix(s1,23);
s2 = getDataBeforeTime(c2, s_time(2));
s2 = getMatrix(s2,23);
s3 = getDataBeforeTime(c3, s_time(3));
s3 = getMatrix(s3,23);
ns1 = getDataBeforeTime(c,700);
ns1 = getMatrix(ns1,23);
ns3 = getDataBeforeTime(c,1400);
ns3 = getMatrix(ns3,23);
ns4 = getDataBeforeTime(c,2100);
ns4 = getMatrix(ns4,23);
ns5 = getDataBeforeTime(c,2800);
ns5 = getMatrix(ns5,23);
ns6 = getDataBeforeTime(c,3500);
ns6 = getMatrix(ns6,23);


s4 = getDataBeforeTime(c4, s_time(4));
s4 = getMatrix(s4,23);
s5 = getDataBeforeTime(c5, s_time(5));
s5 = getMatrix(s5,23);
ns2 = getDataBeforeTime(c,3100);
ns2 = getMatrix(ns2,23);
clear c1 c2 c3 c4 c5 c6 c
clc;
%%
% Processing data
features_s1 = getFeature(s1);
features_s2 = getFeature(s2);
features_s3 = getFeature(s3);
features_s4 = getFeature(s4);
features_ns1 = getFeature(ns1);
features_ns2 = getFeature(ns2);
features_ns3 = getFeature(ns3);
features_ns4 = getFeature(ns4);
features_ns5 = getFeature(ns5);
train_data = [features_s1;features_s2;features_s3;features_ns1;features_s4;features_ns3;features_ns4];
% for test
features_s5 = getFeature(s5);
test_data = [features_s5;features_ns2;features_ns5];
% feature Selection
%%
train_labels = [1,1,1,-1,1,-1,-1]';
test_labels = [1,-1,-1]';
[train_data, test_data] = tTestSelection(train_data,test_data,train_labels,0.001);
%% Classification SVM
svmModel = fitcsvm(train_data,train_labels);
predictedLabels = predict(svmModel, test_data);
%% Classification KNN
k = 3;
knnModel = fitcknn(train_data, train_labels, 'NumNeighbors',k);
predictions_knn = predict(knnModel,test_data);

%% Leave on out

% KNN

data = [train_data;test_data];
labels = [train_labels;test_labels];
trueLabels = labels;
predictedLabels_KNN_l = zeros(10, 1);
TP = 0;
FP = 0;
% Leave-One-Out Cross-Validation
for i = 1:10
    trainingData = data([1:i-1, i+1:end], :);
    trainingLabels = labels([1:i-1, i+1:end]);
    
    testData = data(i, :);
    [trainingData, testData] = tTestSelection(trainingData,testData,trainingLabels,0.01);
    knnModel = fitcknn(trainingData, trainingLabels, 'NumNeighbors', k);
    predictedLabel = predict(knnModel, testData);

    predictedLabels_KNN_l(i) = predictedLabel;
    if predictedLabel == 1 && trueLabels(i) == 1
            TP = TP + 1; % True positive
    elseif predictedLabel == 1 && trueLabels(i) == -1
            FP = FP + 1; % False positive
    end
end

accuracy = sum(predictedLabels_KNN_l == labels) / 10;
TPR = TP / sum(trueLabels == 1); % True Positive Rate (Sensitivity)
FPR = FP / sum(trueLabels == -1); % False Positive Rate

disp(['Accuracy KNN: ', num2str(accuracy)]);
disp(['True Positive Rate (Sensitivity) KNN: ', num2str(TPR)]);
disp(['False Positive Rate KNN: ', num2str(FPR)]);
%%
predictedLabels_SVM_l = zeros(10, 1);
TP = 0;
FP = 0;
% Leave-One-Out Cross-Validation
for i = 1:10
    trainingData = data([1:i-1, i+1:end], :);
    trainingLabels = labels([1:i-1, i+1:end]);
    
    testData = data(i, :);
    [trainingData, testData] = tTestSelection(trainingData,testData,trainingLabels,0.01);
    SVMModel = fitcsvm(trainingData,trainingLabels);
    predictedLabel = predict(SVMModel, testData);

    predictedLabels_SVM_l(i) = predictedLabel;
    if predictedLabel == 1 && trueLabels(i) == 1
            TP = TP + 1; % True positive
    elseif predictedLabel == 1 && trueLabels(i) == -1
            FP = FP + 1; % False positive
    end
end

accuracy = sum(predictedLabels_SVM_l == labels) / 10;
TPR = TP / sum(trueLabels == 1); % True Positive Rate (Sensitivity)
FPR = FP / sum(trueLabels == -1); % False Positive Rate

disp(['Accuracy SVM: ', num2str(accuracy)]);
disp(['True Positive Rate (Sensitivity) SVM: ', num2str(TPR)]);
disp(['False Positive Rate SVM: ', num2str(FPR)]);