
load wordtrain_new_idx.mat
load ./train_set/words_train.mat 
X_label = X;
load ./train_set_unlabeled/words_train_unlabeled.mat

X_unlabel = X;
[X_label_new] = generate_new_wordtrain(topwords_idx, X_label);
[X_unlabel_new] = generate_new_wordtrain(topwords_idx, X_unlabel);
class = 10;
options = statset('MaxIter', 100);
data = [X_label_new; X_unlabel_new];
GMMmodel = fitgmdist(X_label_new,class,'Options',options,'CovarianceType','full','RegularizationValue',1e-5);
idx = cluster(GMMmodel,X_label_new);
C = GMMmodel.mu;
trans = zeros(class,1);
for i = 1:class
    trans(i) = mode(Y(idx == i));
end

label = zeros(size(X_label_new,1),1);

for i = 1:size(X_label_new,1)
    dis = zeros(class,1);
    for j = 1:class
        dis(j) = norm(X_label_new(i,:)-C(j,:));
    end
    [~,index] = min(dis);
    label(i) = trans(index);
end

mean(label == Y)
