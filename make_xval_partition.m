function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% UPDATED SEP 16 2016

% YOUR CODE GOES HERE
rest = rem(n,n_folds);
n_onefold = (n - rest)/n_folds; % number of datapoints in each fold

% allocate a random value in r(i) for sample i (0 < r(i) < 1)
r = rand(n,1);

% sort these random values
[sorted_r,ind] = sort(r);

part = zeros(1,n);


for f = 1 : n_onefold
    part(ind((f-1)*n_folds+1:f*n_folds)) = (1:n_folds)';
end

if rest > 0
    part(ind(n_folds*n_onefold+1:n)) = (1:rest)';
end


end