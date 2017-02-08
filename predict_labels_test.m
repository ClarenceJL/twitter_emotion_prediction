function [Y_hat, accuracy, prob] = predict_labels_test(X_train,Y_train,X_test,Y_test,method, reform_data, param)



% data preprocessing
if reform_data
    [X_train,X_test] = preprocess_data2(X_train,X_test);  % rescale image
end


switch method
    % method 1: KNN
    case 'KNN'   % default
        disp('KNN..')
        Y_train = Y_train * 2 - 1; Y_test = Y_test * 2 - 1;
        % use the MATLAB built-in knn   
        model = fitcknn(X_train,Y_train,'NumNeighbors',200);
        save('color_knn.mat','model');
        [Y_hat,prob] = predict(model,X_test);
        %
        accuracy = mean(Y_hat == Y_test);
        Y_hat = (Y_hat + 1) / 2; % 0, 1
        %prob = []; 
        %Y_hat = preval2prelabel(Y_test_hat);
        
    % method 2: Kernel Regression
    case 'KerReg' 
        disp('Kernel Regression ...');
        Y_train = Y_train * 2 - 1; Y_test = Y_test * 2 - 1;
        sigma = 15;
        Kernel = kernel_gaussian(X_train, X_test, sigma);
        % Kernel = kernel_intersection(X_train, X_test);   
        % Kernel = kernel_poly(X_train,X_test,2);
        Y_hat = sign(Kernel * Y_train);
        accuracy = mean(Y_hat == Y_test);
        Y_hat = (Y_hat + 1) / 2; % 0, 1
        % Y_hat = preval2prelabel(Y_test_hat);
        
        
    case 'SVM'
        disp('SVM...')
        addpath ./libsvm
        % method 1: using libsvm
        Y_train = Y_train*2 - 1;
        Y_test = Y_test*2 - 1;
        ker = @(x,x2) kernel_intersection(x, x2);
        %ker = @(x,x2) kernel_poly(x, x2, 1);
        % sigma = 20;
        % ker = @(x,x2) kernel_gaussian(x,x2,sigma);
        [Y_hat, accuracy, prob] = libsvm_ker(X_train, Y_train, X_test, Y_test, ker,1);
        
        Y_hat = (Y_hat + 1) / 2;
        
    
    case 'logistic' % logistic regression
        disp('Logistic regression...')
        
        addpath ./liblinear
        
        if nargin < 7
            param = 2;
        end
        if param == 2
            model = train(Y_train, sparse(X_train), ['-s 0', 'col']);
            %save('word_logistic.mat','model');
        elseif param == 1
            model = train(Y_train, sparse(X_train), ['-s 6', 'col']);
            %save('cnn_logistic.mat','model');
        end
        
        %save('word_logistic.mat','model');
        [Y_hat, ~, prob] = predict(Y_test, sparse(X_test), model, ['-b 1', '-q', 'col']);
        
        accuracy = mean(Y_hat == Y_test);
        
    case 'NB'  % Naive Bayes
        disp('Naive Bayes...')
        Y_train = Y_train + 1;
        B = fitcnb(X_train,Y_train, 'Distribution','mvmn');
        %save('word_nb.mat','B');
        [Y_hat,prob] = predict(B,X_test);

        Y_hat = round(Y_hat);
        Y_hat(Y_hat < 1 ) = 1; 
        Y_hat(Y_hat > 2) = 2;  
        
        accuracy = mean(Y_hat == Y_test);
        Y_hat = Y_hat - 1; % 1,0 
        
    case 'DT'   % decision tree
        disp('Decision tree...')
        depth_limit = 16;
        tree = dt_train(X_train, Y_train, depth_limit);
        Y_hat = zeros(size(Y_test));
        for i  = 1:length(Y_test)
            Y_hat(i) = dt_value(tree,X_test(i,:));
        end
        Y_hat = round(Y_hat);
        accuracy = mean(Y_hat == Y_test);
        
        
    case 'forest' % random forest
        disp('Random forest...')
        if nargin < 7
            param = [300 0.3];
        end
        model = TreeBagger(param(1), X_train, Y_train,'InBagFraction',param(2));
        save('word_forest.mat','model');
        [Y_hat,prob] = predict(model,X_test);
        Y_hat = cell2mat(Y_hat);
        Y_hat = double(Y_hat - '0');
        accuracy = mean(Y_hat == Y_test);
        
    case 'perceptron' % perceptron
        net = perceptron;
        net = train(net,X_trian,Y_train);
        Y_hat = net(X_test);
        accuracy = mean(Y_hat == Y_test);
        
    case 'NN' % neural network
        %neural net with L2 weight decay
        rand('state',0)
        % [784 = input dimension, 
        % 100 = size of hidden layer, 
        % 10 = size of output] 
        nn = nnsetup([size(X_train,2) 100 2]); 
        nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
        opts.numepochs =  20;       %  Number of full sweeps through data
        opts.batchsize = 100;       %  Take a mean gradient step over this many samples

        [nn, loss] = nntrain(nn, X_train, Y_train, opts);

        [er, bad] = nntest(nn, X_test, Y_test);
        Y_hat = Y_test;
        Y_hat(bad) = 1 - Y_hat(bad);
        
        accuracy = 1 - er;
        
    case 'boosting'
        disp('Boosting...')
        model = fitensemble(full(X_train),full(Y_train),'LogitBoost',500,'Tree');
        save('word_boosting.mat','model');
        [Y_hat, prob] = predict(model, full(X_test));
        accuracy = mean(Y_hat == Y_test);   
        
end



end


