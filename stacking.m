function [ stacking_model, train_acc ] = stacking(part,i, X, Y, train_color, train_img_prob, train_cnn_feat, train_img, raw_tweets_train)
% INPUT
% X_train, Y_train - data to train different models
% X_test - data to test the models above, and the estimated Y_test_hat are
% used together with real label Y_test to train a ensemble model
% OUTPUT
% model - a logistic regression or SVM model to combine all results
% together




ind_train = (part ~= i );
ind_test = (part == i );
Y_train = Y(ind_train);
Y_test = Y(ind_test);
Z_test_hat = [];

% train logistic regression model for word counts
disp('Model 1...');
X_train = X(ind_train,:); 
X_test = X(ind_test,:);
[Y_hat,acc,prob] = predict_labels_test(X_train,Y_train,X_test,Y_test,'logistic',0);
disp(['logistic regression for words, accuracy:' num2str(mean(Y_hat == Y_test))]);
Z_test_hat(:,1) = prob(:,1);


% train random forest model for image cnn feature
disp('Model 2...');
X_train = train_cnn_feat(ind_train,:);
X_test = train_cnn_feat(ind_test,:);
[Y_hat,acc,prob] = predict_labels_test(X_train,Y_train,X_test,Y_test,'forest',1);
disp(['Random forest for CNN feature, accuracy:' num2str(acc) ])
Z_test_hat(:,2) = prob(:,2);


% train knn model for image color feature
% disp('Model 3...');
% X_train = train_color(ind_train,:);
% X_test = train_color(ind_test,:);
% [Y_hat,acc,prob] = predict_labels_test(X_train, Y_train,X_test,Y_test,'KNN',0);
% disp(['KNN for color feature, accuracy:' num2str(acc) ])
% Z_test_hat(:,3) = prob(:,2);

% train ... model for image probability feature
% disp('Model 4...')
% X_train = train_img_prob(ind_train,:);
% X_test = train_img_prob(ind_test,:);
% [Y_hat,acc,prob] = predict_labels_test(X_train,Y_train,X_test,Y_test,'SVM',1);
% disp(['SVM for image probability, accuracy:' num2str(mean(Y_hat == Y_test))]);
% Z_test_hat(:,4) = prob(:,1);


ensemble_method = 'ridge';

switch ensemble_method
    case 'logistic'
        disp('ensembling...')
        addpath ./liblinear
        stacking_model = train(full(Y_test), sparse(Z_test_hat), ['-s 0', 'col']);
        save('stacking_model_logistic.mat',stacking_model);
        [Y_final_hat] = predict(full(Y_test), sparse(Z_test_hat), stacking_model, [ '-q', 'col']);
        train_acc = mean(Y_final_hat == Y_test) ;
        
    case 'ridge'
        disp('ensembling with linear regression...');
        lambda = 0.1 ;
        stacking_model = ridge(Y_test, Z_test_hat, lambda,0);
        save('stacking_model_ridge.mat',stacking_model);
        [Y_final_hat] = round ([ones(size(Y_test)) Z_test_hat] * stacking_model);
        train_acc = mean(Y_final_hat == Y_test);
        
    case 'lasso'
        alpha = 1;
        stacking_model = lasso(Z_test_hat, Y_test);
        
    case 'forest'
        stacking_model = TreeBagger(300, Z_test_hat, Y_test,'InBagFraction',0.3);
        Y_hat = predict(stacking_model,Z_test_hat);
        Y_final_hat = double(cell2mat(Y_hat) - '0');
        train_acc = mean(Y_final_hat == Y_test);
        
    
end
