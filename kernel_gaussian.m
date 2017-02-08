function K = kernel_gaussian(X, X2, sigma)
% Evaluates the Gaussian Kernel with specified sigma
%
% Usage:
%
%    K = KERNEL_GAUSSIAN(X, X2, SIGMA)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X2(i,:), X(j,:)) and k is the Guassian kernel
% with parameter sigma=20.


n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.
X = X'; X2 = X2';


for j = 1:n
    diff = bsxfun(@minus,X2,X(:,j));   % D*m
    diff = diff.^2;
    diff = sum(diff)./(2*sigma^2);     % 1*m
    K(:,j) = exp(-diff)';               % m*1
end


