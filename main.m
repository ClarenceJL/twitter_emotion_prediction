%% main

clear
clc

%% library path
% addpath(genpath('../stprtool/'));
addpath ./decisiontree
addpath(genpath('./DL_toolbox/'));

%% load training data
load ./train_set/words_train.mat       % X(sparse), Y(sparse), tweet_ids 
load ./train_set/train_cnn_feat.mat    % train_cnn_feat
load ./train_set/train_color.mat       % train_color
load ./train_set/train_img_prob.mat    % train_img_prob
load ./train_set/train_raw_img.mat     % train_img
load ./train_set/train_tweet_id_img.mat% train_tweet_id_img
load ./train_set/raw_tweets_train.mat  % raw_tweets_train(cell)
load words_train_stemming.mat          % X_new, topwords_idx
%load ./train_set/train_HOG.mat         % train_HOG

%% word stemming


%% split data
n = length(Y);
Y = full(Y);
part = make_xval_partition(n, 9);
accuracy = zeros(9,1);


%% stacking
% for i = 8:8
%     disp(['Batch ' num2str(i)]);
%     [stacking_model, train_acc(i),] = all_stacking(part,i, X_new, Y, train_color, train_img_prob, train_cnn_feat, train_img, raw_tweets_train);
% end

disp('pca...')
loadings = pca(train_cnn_feat);     % pca automatically standarize data
% % standarize trainset
mean_train = mean(train_cnn_feat);
data_train = bsxfun(@minus,train_cnn_feat,mean_train);
% % calculate the principle components of trainset
train_score = data_train * loadings;

% data = [X train_color];

%% Cross Validation 
for i = 1:9
    tic
    disp(['Batch ' num2str(i)]);
    
    Y_train = Y((part ~= i));
    Y_test = Y((part == i));
    
    % word
    X_train = train_score((part ~= i),1:500);
    X_test = train_score((part == i),1:500);
    
    
    [Y_test_hat, ~ ,prob] = predict_labels_test(X_train, Y_train, X_test, Y_test,'forest',0);
    accuracy(i) = mean(Y_test_hat == Y_test);
    toc
    
end

%Y_test_hat = predict_labels_test(train_HOG,Y,train_HOG,Y);
%accuracy = mean(Y == Y_test_hat);


