function result = getDataBeforeTime(edfData, targetTime)
result = edfData(targetTime-604:targetTime-61,:);
end