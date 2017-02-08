%% call predict_labels.m

load ./train_set/words_train.mat       % X(sparse), Y(sparse), tweet_ids 
load ./train_set/train_cnn_feat.mat    % train_cnn_feat
load ./train_set/train_color.mat       % train_color
load ./train_set/train_img_prob.mat    % train_img_prob
load ./train_set/train_raw_img.mat     % train_img
load ./train_set/train_tweet_id_img.mat% train_tweet_id_img
load ./train_set/raw_tweets_train.mat  % raw_tweets_train (cell)
load words_train_stemming.mat          % X_new, topwords_idx

[Y_hat] = predict_labels(X, train_cnn_feat, train_img_prob, train_color, train_img, raw_tweets_train);

accuracy = mean(Y_hat == Y);