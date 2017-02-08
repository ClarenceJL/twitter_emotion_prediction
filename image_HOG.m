function [model, HOG_train, Ypred_svm, Yprob_svm] = image_HOG(Xtrain, Ytrain, Xtest, Ytest)
cell_size = 7;
Ntrain = size(Xtrain, 1);
Ntest = size(Xtest, 1);

hog_size = 1296;
HOG_train = zeros(Ntrain, hog_size);
HOG_test = zeros(Ntest, hog_size);

for i = 1:Ntrain
    I = reshape(Xtrain(i, :), [50, 50, 3]);
    I = double(I);
    % EXTRACT HOG FEATURES (MATLAB PROVIDED)
    [hog_8x8, ~] = extractHOGFeatures(I,'CellSize',[cell_size cell_size]);
    HOG_train(i, :) = hog_8x8;
end

for i = 1:Ntest
    I = reshape(Xtest(i, :), [50, 50, 3]);
    I = double(I);
    [hog_8x8, ~] = extractHOGFeatures(I,'CellSize',[cell_size cell_size]);
    HOG_test(i, :) = hog_8x8;
end

[model, Ypred_svm, Yprob_svm] = svm_img(HOG_train, Ytrain, HOG_test, Ytest);
sum(Ypred_svm == Ytest)/size(Ytest, 1)

% PARAMETERS NEED TO BE DECIDED AGAIN USING CV
% ANOTHER SET OF PARAMETERS: 9/10 0.05 8

% [Ypred_svm, Yprob_svm] = svm_img(HOG_train, Ytrain, HOG_test, Ytest, 1);
% correct_ratio = sum(Ypred_svm == Ytest)/size(Ytest, 1)
end

