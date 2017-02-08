function [ stacking_model, train_acc ] = all_stacking(part,i, X, Y, train_color, train_img_prob, train_cnn_feat, train_img, raw_tweets_train)
%% Train multiple models for emotion classification and ensemble
% INPUT
% X_train, Y_train - data to train different models
% X_test - data to test the models above, and the estimated Y_test_hat are
% used together with real label Y_test to train a ensemble model
% OUTPUT
% model - a logistic regression or SVM model to combine all results
% together

%% split data
ind_train = (part ~= i );
ind_test = (part == i );
Y_train = Y(ind_train);
Y_test = Y(ind_test);
Z_test_word = [];


%% Part 1: ensemble 5 word-based models
X_train = X(ind_train,:); 
X_test = X(ind_test,:);

% train logistic regression model for word count
disp('Model 1 ...');
[Y_hat, ~, pro] = predict_labels_test(X_train,Y_train,X_test,Y_test,'logistic',0);
disp(['logistic regression for words, accuracy:' num2str(mean(Y_hat == Y_test))]);
Z_test_word = [Z_test_word pro];

% train random forest model for word count
disp('Model 2 ...');
[Y_hat,~, pro] = predict_labels_test(full(X_train),Y_train,full(X_test),Y_test,'forest',0,[300 0.3]);
disp(['Random forest for words, accuracy:' num2str(mean(Y_hat == Y_test)) ])
Z_test_word = [Z_test_word pro];

% train boosting model for word count
disp('Model 3 ...')
[Y_hat,~, pro] = predict_labels_test(X_train,Y_train,X_test,Y_test,'boosting',0);
disp(['Boosting for words, accuracy:' num2str(mean(Y_hat == Y_test)) ])
Z_test_word = [Z_test_word pro];

% train naive bayes model for word count
disp('Model 4 ...');
[Y_hat,~, pro] = predict_labels_test(X_train,Y_train,X_test,Y_test,'NB',0);
disp(['NB for words, accuracy:' num2str(mean(Y_hat == Y_test)) ])
Z_test_word = [Z_test_word pro];

% train SVM model for word count
disp('Model 5 ...');
[Y_hat,~, pro] = predict_labels_test(X_train,Y_train,X_test,Y_test,'SVM',0);
disp(['SVM for words, accuracy:' num2str(mean(Y_hat == Y_test)) ])
Z_test_word = [Z_test_word pro];

% ensembling
Z_test_word = Z_test_word(:,1:2:10);

ensemble_method = 'logistic';
[stacking_model, train_acc, prob_word] = ensemble_models(Z_test_word, Y_test, ensemble_method);
disp(['ensembling for words, accuracy: ' num2str(train_acc)])


%% Part 2: ensemble word models with word and image

% ensembled word count prediction
Z_test_all(:,1) = prob_word(:,1);


% train random forest model for image cnn feature
disp('Model 6...');
X_train = train_cnn_feat(ind_train,:);
X_test = train_cnn_feat(ind_test,:);
[Y_hat,acc,prob] = predict_labels_test(full(X_train),Y_train,full(X_test),Y_test,'logistic',1,1);
disp(['Logistic for CNN feature, accuracy:' num2str(acc) ])
Z_test_all(:,2) = prob(:,2);


% train logistic model for image color feature
% disp('Model 7...');
X_train = train_color(ind_train,:);
X_test = train_color(ind_test,:);
[Y_hat,acc,prob] = predict_labels_test(X_train, Y_train,X_test,Y_test,'KNN',0);
disp(['Logistic for color feature, accuracy:' num2str(acc) ])
Z_test_all(:,3) = prob(:,2);

% train ... model for image probability feature
% disp('Model 4...')
% X_train = train_img_prob(ind_train,:);
% X_test = train_img_prob(ind_test,:);
% [Y_hat,acc,prob] = predict_labels_test(X_train,Y_train,X_test,Y_test,'SVM',1);
% disp(['SVM for image probability, accuracy:' num2str(mean(Y_hat == Y_test))]);
% Z_test_hat(:,4) = prob(:,1);


ensemble_method = 'ridge';
[stacking_model, train_acc, ~] = ensemble_models(Z_test_all, Y_test, ensemble_method);
disp(['ensembling for all, accuracy: ' num2str(train_acc)])


end


%% 
function [stacking_model, train_acc, prob] = ensemble_models(Z_test, Y_test, method, lambda)

switch method
    case 'logistic'
        disp('ensembling...')
        addpath ./liblinear
        stacking_model = train(full(Y_test), sparse(Z_test), ['-s 0', 'col']);
        save('stacking_model_logistic.mat','stacking_model');
        [Y_final_hat,~,prob] = predict(full(Y_test), sparse(Z_test), stacking_model, ['-b 1' '-q', 'col']);
        train_acc = mean(Y_final_hat == Y_test) ;
        
    case 'SVM'
        disp('ensembling...')
        addpath ./libsvm
        ker = @(x,x2) kernel_intersection(x, x2);
        [Y_final_hat] = libsvm_ker(X_train, Y_train, X_test, Y_test, ker);   
        
    case 'ridge'
        disp('ensembling with linear regression...');
        if nargin < 4
            lambda = 0.1 ;
        end
        stacking_model = ridge(Y_test, Z_test, lambda,0);
        save('stacking_model_ridge.mat','stacking_model');
        [Y_final_hat] = round ([ones(size(Y_test)) Z_test] * stacking_model);
        
        train_acc = mean(Y_final_hat == Y_test);
        prob = [];
        
    case 'lasso'
        alpha = 1;
        stacking_model = lasso(Z_test_hat, Y_test);
        
    
end

end
