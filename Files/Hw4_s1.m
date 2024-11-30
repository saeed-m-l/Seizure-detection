%% Part 1 ( ready file -- preapare data) - subject 1
Fs = 256;
% For sub 1 s in [3,4,15,16,18,26]
clear;close all;clc;
c1 = edfread("Sub1\WS1\chb01_03.edf");
c2 = edfread("Sub1\WS1\chb01_04.edf");
c3 = edfread("Sub1\WS1\chb01_15.edf");
c4 = edfread("Sub1\WS1\chb01_16.edf");
c5 = edfread("Sub1\WS1\chb01_18.edf");
c6 = edfread("Sub1\WS1\chb01_26.edf");
c = edfread("Sub1\WOS1\chb01_02.edf");
s_time = [2996,1476,1732,1015,1720,1862];
s1 = getDataBeforeTime(c1, s_time(1));
s1 = getMatrix(s1,23);
s2 = getDataBeforeTime(c2, s_time(2));
s2 = getMatrix(s2,23);
s3 = getDataBeforeTime(c3, s_time(3));
s3 = getMatrix(s3,23);
ns1 = getDataBeforeTime(c,800);
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
s6 = getDataBeforeTime(c6, s_time(6));
s6 = getMatrix(s6,23);
ns2 = getDataBeforeTime(c,1600);
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
features_ns3 = getFeature(ns3);
features_ns4 = getFeature(ns4);
features_ns5 = getFeature(ns5);
features_ns6 = getFeature(ns6);
tra_data = [features_s1;features_ns3;features_s2;features_ns4;features_s3;features_ns1;features_s4;features_ns5];
% for test
features_s5 = getFeature(s5);
features_s6 = getFeature(s6);
features_ns2 = getFeature(ns2);
tes_data = [features_s5;features_ns6;features_s6;features_ns2];
% feature Selection
%%
train_labels = [1,-1,1,-1,1,-1,1,-1]';
test_labels = [1,-1,1,-1]';
[train_data, test_data] = tTestSelection(tra_data,tes_data,train_labels,0.01);
%% Classification SVM
svmModel = fitcsvm(train_data,train_labels);
predictedLabels = predict(svmModel, test_data);
%% Classification KNN
k = 3;
knnModel = fitcknn(train_data, train_labels, 'NumNeighbors',k);
predictions_knn = predict(knnModel,test_data);
%% Leave on out

% KNN

data = [tra_data;tes_data];
labels = [train_labels;test_labels];
trueLabels = labels;
predictedLabels_KNN_l = zeros(12, 1);
TP = 0;
FP = 0;
% Leave-One-Out Cross-Validation
for i = 1:12
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

accuracy = sum(predictedLabels_KNN_l == labels) / 12;
TPR = TP / sum(trueLabels == 1); % True Positive Rate (Sensitivity)
FPR = FP / sum(trueLabels == -1); % False Positive Rate

disp(['Accuracy KNN: ', num2str(accuracy)]);
disp(['True Positive Rate (Sensitivity) KNN: ', num2str(TPR)]);
disp(['False Positive Rate KNN: ', num2str(FPR)]);
%%
predictedLabels_SVM_l = zeros(12, 1);
TP = 0;
FP = 0;
% Leave-One-Out Cross-Validation
for i = 1:12
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

accuracy = sum(predictedLabels_SVM_l == labels) / 12;
TPR = TP / sum(trueLabels == 1); % True Positive Rate (Sensitivity)
FPR = FP / sum(trueLabels == -1); % False Positive Rate

disp(['Accuracy SVM: ', num2str(accuracy)]);
disp(['True Positive Rate (Sensitivity) SVM: ', num2str(TPR)]);
disp(['False Positive Rate SVM: ', num2str(FPR)]);