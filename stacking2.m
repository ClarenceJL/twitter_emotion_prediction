% [ stacking_model, train_acc ] = stacking(part, X, Y, train_color, train_img_prob, train_cnn_feat, train_img, raw_tweets_train)
% INPUT
% X_train, Y_train - data to train different models
% X_test - data to test the models above, and the estimated Y_test_hat are
% used together with real label Y_test to train a ensemble model
% OUTPUT
% model - a logistic regression or SVM model to combine all results
% together
%load_data
%X = X_new;

load ./train_set/words_train.mat

n = length(Y);
part = make_xval_partition(n, 9);

ind_train = (part ~= 3);
ind_test = (part == 3);
Y_train = Y(ind_train);
Y_test = Y(ind_test);
Z_test_hat = [];

% data_test = tweet_ids(part == 3);
% raw_data = raw_tweets_train{1,2}(part == 3);

% train logistic regression model for word counts
disp('Model1...');
X_train = X(ind_train,:); 
X_test = X(ind_test,:);


[Y_hat, ~, pro] = predict_labels_test(X_train,Y_train,X_test,Y_test,'logistic');
disp(['logistic regression for words, accuracy:' num2str(mean(Y_hat == Y_test))]);
Z_test_hat = [Z_test_hat pro];


% train random forest model for image cnn feature
disp('Model2...');
% X_train = train_cnn_feat(ind_train,:);
% X_test = train_cnn_feat(ind_test,:);
% [Y_hat,~] = predict_labels_test(X_train,Y_train,X_test,Y_test,'forest');
% disp(['Random forest for CNN feature, accuracy:' num2str(mean(Y_hat == Y_test)) ])
% Z_test_hat(:,2) = Y_hat;

[Y_hat,~, pro] = predict_labels_test(X_train,Y_train,X_test,Y_test,'forest');
disp(['Random forest for words, accuracy:' num2str(mean(Y_hat == Y_test)) ])
Z_test_hat = [Z_test_hat pro];

% train knn model for image color feature
disp('Model3...');
% X_train = train_color(ind_train,:);
% X_test = train_color(ind_test,:);
% [Y_hat,~] = predict_labels_test(X_train, Y_train,X_test,Y_test,'KNN');
% disp(['KNN for color feature, accuracy:' num2str(mean(Y_hat == Y_test)) ])
% Z_test_hat(:,3) = Y_hat;

[Y_hat,~, pro] = predict_labels_test(X_train,Y_train,X_test,Y_test,'boosting');
disp(['Boosting for words, accuracy:' num2str(mean(Y_hat == Y_test)) ])
Z_test_hat = [Z_test_hat pro];

disp('Model4...');
[Y_hat,~, pro] = predict_labels_test(X_train,Y_train,X_test,Y_test,'NB');
disp(['NB for words, accuracy:' num2str(mean(Y_hat == Y_test)) ])
Z_test_hat = [Z_test_hat pro];


disp('Model5...');
[Y_hat,~, pro] = predict_labels_test(X_train,Y_train,X_test,Y_test,'SVM');
disp(['SVM for words, accuracy:' num2str(mean(Y_hat == Y_test)) ])
Z_test_hat = [Z_test_hat pro];

save ('ensembledata', 'Z_test_hat')

ensemble_method = 'logistic';

switch ensemble_method
    case 'logistic'
        disp('ensembling...')
        addpath ./liblinear
        stacking_model = train(Y_test, sparse(Z_test_hat), ['-s 0', 'col']);
        [Y_final_hat] = predict(Y_test, sparse(Z_test_hat), stacking_model, ['-q', 'col']);
        train_acc = mean(Y_final_hat == Y_test);
        
    case 'SVM'
        disp('ensembling...')
        addpath ./libsvm
        ker = @(x,x2) kernel_intersection(x, x2);
        [Y_final_hat] = kernel_libsvm(X_train, Y_train, X_test, Y_test, ker);
        
    case 'forest'
        stacking_model = TreeBagger(100, Z_test_hat, Y_test,'InBagFraction',0.5);
        Y_hat = predict(stacking_model,Z_test_hat);
        Y_final_hat = double(cell2mat(Y_hat) - '0');
        train_acc = mean(Y_final_hat == Y_test);
       
end
