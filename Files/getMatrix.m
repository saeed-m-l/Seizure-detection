function newMatrix = getMatrix(cellArray, column)
row = 544;
newMatrix = zeros(256*row, column);
for i = 1:size(cellArray, 1)
    for j = 1:size(cellArray, 2)
        % Get the matrix from the current cell
        currentMatrix = cell2mat(cellArray{i, j});
        
        % Reshape the matrix to a column vector and assign it to the new matrix
        newMatrix(((i-1)*256)+1:i*256, j) = currentMatrix(:);
    end
end
newMatrix= rmmissing(newMatrix,2);
end