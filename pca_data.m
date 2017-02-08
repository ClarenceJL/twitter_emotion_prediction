function [ train_data, test_data ] = pca_data( x_train, x_test, numpc )
% use PCA to reduce data dimension
% INPUT
% x_train - training data
% x_test - test data
% numpc - number of principle components selected
% OUTPUT
% train_data - PCA-ed training data
% TEST_DATA - PCA-ed test data


%% pca training
loadings = pca(x_train);     % pca automatically standarize data
% standarize trainset
mean_train = mean(x_train);
data_train = bsxfun(@minus,x_train,mean_train);
% calculate the principle components of trainset
pcscore_train = data_train * loadings;

%% map test data on the loadings
% standarize testset
data_test = bsxfun(@minus,x_test,mean_train);
% calculate the principle components of testset
pcscore_test = data_test * loadings;


%% reduce data dimension by selecting numpc top principle components
if nargin < 3
    % dynamically select numpc
    [~,p] = size(loadings);
    step = ceil(p/30);
    for i = 1:step:p
        x_train_rec = pcscore_train(:,1:i) * (loadings(:,1:i))';
        rec_accuracy = sum(sum(x_train_rec.^2)) / sum(sum(data_train.^2));
        if rec_accuracy >= 0.9
            numpc = i;
            break;
        end
    end
end

train_data = pcscore_train(:,1:numpc);
test_data = pcscore_test(:,1:numpc);

end

