function [ X_new] = preprocess_data1( X )
%% reform data using one of the following techniques:
% Transformation:
% - scaling (default)
% - normalization
% - standarization
% - centering
% Reduction:
% - sampling
% - PCA
% - auto-encoder

% Reference: http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html

if nargin < 2
    tech = 'scale';
end

n_train = size(X,1);
mean_X = mean(X);
std_X = std(X);
max_X = max(X);
min_X = min(X);

lambda = 1e-6;

switch tech
    case 'scale'
        X_new = (bsxfun(@minus,X,min_X)) ./ (bsxfun(@minus,max_X,min_X) + lambda);
        
    case 'norm'
       
    case 'center'
         X_new = bsxfun(@minus,X,mean_X);
         
    case 'standard'
         X_new = bsxfun(@minus,X,mean_X);
         X_new = bsxfun(@ldivide,X_new,std_X+lambda);

    case 'sample'
        
    case 'PCA'
        
    case 'autoencoder'
        
end





end

