function [ X_train_new, X_test_new ] = preprocess_data2( X_train, X_test, tech )
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

if nargin < 3
    tech = 'scale';
end

n_train = size(X_train,1);
n_test = size(X_test,1);

mean_train = mean(X_train); % Mean_train = repmat(mean_train, n_train, 1);
std_train = std(X_train);
max_train = max(X_train);
min_train = min(X_train);

lambda = 1e-6;

switch tech
    case 'scale'
        
        X_train_new = (X_train-repmat(min_train, n_train, 1)) ./ (repmat(max_train-min_train,n_train,1) + lambda);
        X_test_new = (X_test-repmat(min_train, n_test, 1)) ./ (repmat(max_train-min_train,n_test,1) + lambda);
    
    case 'norm'
       
    case 'center'
         X_train_new = bsxfun(@minus,X_train,mean_train);
         X_test_new = bsxfun(@minus,X_test,mean_train);
         
    case 'standard'
         X_train_new = bsxfun(@minus,X_train,mean_train);
         X_train_new = bsxfun(@ldivide,X_train_new,std_train+lambda);
         X_test_new = bsxfun(@minus,X_test,mean_train);       
         X_test_new = bsxfun(@ldivide,X_test_new,std_train+lambda);
        
    case 'sample'
        
    case 'PCA'
        [ X_train_new, X_test_new ] = pca_data(full(X_train), full(X_test));
        
    case 'autoencoder'
        
end





end

