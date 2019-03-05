% Shraddheya, Rajeshwari, Joshi, Vanita, Percy
% Actual data from child_500 parent's code only
% data = [61 63 68 57 70;
%         18 28 24 43 20;
%         21  9  8  0 10];

data = [61 63 68 57 70;
        18 28 24 43 20;
        21  9  8  0 10;
        40 33 47 68 44;
        47 67 53 32 50;
        13  0  0  0  6];


for i=1:4
    DATA = data(:, [i 5]);
    [~, tbl, ~] = anova2(DATA, 1, 'off');
    MSC = tbl{2, 4};
    MSR = tbl{3, 4};
    MSE = tbl{4, 4};
    [n, k] = size(DATA);
    ICC = (MSR - MSE) / (MSR + MSE * (k - 1) + (k / n) * (MSC - MSE))
end
