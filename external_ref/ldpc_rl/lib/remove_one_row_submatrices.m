function C = remove_one_row_submatrices(A)
    m = size(A,1);
    C = cell(m,1);
    
    for i = 1:m
        C{i} = A([1:i-1, i+1:end], :);
    end
end
