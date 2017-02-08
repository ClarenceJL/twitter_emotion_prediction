function [Y_hat] = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets)
% Inputs:   word_counts     nx10000 word counts features
%           cnn_feat        nx4096 Penultimate layer of Convolutional
%                               Neural Network features
%           prob_feat       nx1365 Probabilities on 1000 objects and 365
%                               scene categories
%           color_feat      nx33 Color spectra of the images (33 dim)
%           raw_imgs        nx30000 raw images pixels
%           raw_tweets      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 0 for sad)


addpath ./libsvm
addpath ./liblinear


%% load training data (if necessary) and models
load ./words_train_stemming.mat
load ./train_set/words_train.mat
load ./train_set/train_cnn_feat.mat 

% word model
load ./word_boosting.mat
w_boosting = model;
load ./word_forest.mat
w_forest = model;
load ./word_logistic.mat
w_logistic = model;
load ./word_nb.mat
w_nb = B;
load ./word_SVM.mat
w_svm = model;

% image model
load ./cnn_logistic.mat
cnn_logistic = model;
load ./color_knn.mat
color_knn = model;

% stacking model
load ./stacking_model_logistic.mat
stacking_logistic = stacking_model;
load ./stacking_model_ridge.mat
stacking_ridge = stacking_model;


%% test on 5 word models
n = size(word_counts,1);      % number of test samples
X_test = generate_new_wordtrain(topwords_idx, word_counts);
Y_test = ones(n,1);
pro_test = [];

disp('word logistic.....')
[Y_hat, ~, prob] = predict((Y_test), (X_test), w_logistic, ['-q -b 1', 'col']);
pro_test = [pro_test prob];
disp('word random forest.....')
[Y_hat, prob] = predict(w_forest,full(X_test));
pro_test = [pro_test prob];
disp('word boosting.......')
[Y_hat, prob] = predict(w_boosting, full(X_test));
pro_test = [pro_test prob];
disp('word NB.........')
[Y_hat, prob] = predict(w_nb,full(X_test));
pro_test = [pro_test prob];
disp('word SVM.......')
kernel = @(x,x2) kernel_intersection(x, x2);
Ktest = kernel(X_new, X_test);
[Y_hat, ~, prob] = svmpredict(Y_test, [(1:size(Ktest,1))' Ktest], w_svm, '-b 1');
pro_test = [pro_test prob];

Z_test = pro_test(:,1:2:10);

[Y_hat, ~, prob_word] = predict((Y_test), sparse(Z_test), stacking_logistic, ['-q -b 1', 'col']);


%% combine word with image
Z_test_all(:,1) = prob_word(:,1);

disp('cnn logistic ...')
[~,cnn_feat] = preprocess_data2(train_cnn_feat,cnn_feat);
[~, ~, prob] = predict(sparse(Y_test), sparse(cnn_feat), cnn_logistic, ['-q -b 1', 'col']);
Z_test_all(:,2) = prob(:,2);

disp('color knn ...')
[~,prob] = predict(color_knn,color_feat);
Z_test_all(:,3) = prob(:,2);


% Final result
Y_hat = round ([ones(n,1) Z_test_all] * stacking_ridge);


end
