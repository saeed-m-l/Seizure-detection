function [features] = getFeature(eegData)
Fs = 256;
epochDuration = 16;
samplesPerEpoch = epochDuration * Fs;
%windowSize = epochDuration * Fs;
numEpochs = floor(size(eegData, 1) / (Fs*epochDuration));
numChannels = size(eegData, 2);
numFeatures = 5;
features = zeros(numEpochs, numChannels * numFeatures);
for i = 1:numEpochs
    startIdx = (i - 1) * samplesPerEpoch + 1;
    endIdx = startIdx + samplesPerEpoch - 1;
    epochSignal = eegData(startIdx:endIdx, :);
    % epochSignal = abs(fft(epochSignal));

    %  pwelch(epochSignal, windowSize, [], [], Fs);
    psd = periodogram(epochSignal);
    psd = abs(psd);
    normalizedPSD = psd./mean(psd);

    entropyValue = -sum(normalizedPSD .* log(normalizedPSD))/log(16);

    average = mean(epochSignal);
    stdDev = std(epochSignal);
    minValue = min(epochSignal);
    maxValue = max(epochSignal);

    features(i, :) = [
        entropyValue, ... 
        average, stdDev, minValue, maxValue 
    ];
end
b = size(features);
features = reshape(features,[b(1)*b(2),1])';
end