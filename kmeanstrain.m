load wordtrain_new_idx.mat
load ./train_set/words_train.mat 
X_label = X;
load ./train_set_unlabeled/words_train_unlabeled.mat
X_unlabel = X;
[X_label_new] = generate_new_wordtrain(topwords_idx, X_label);
[X_unlabel_new] = generate_new_wordtrain(topwords_idx, X_unlabel);
cluster = 20;
[idx,C] = kmeans(X_label,cluster,'MaxIter',100);
trans = zeros(cluster,1);
for i = 1:cluster
    trans(i) = mode(Y(idx == i));
end

% label = zeros(size(X_unlabel_new,1),1);
% 
% for i = 1:size(X_unlabel_new,1)
%     dis = zeros(cluster,1);
%     for j = 1:cluster
%         dis(j) = norm(X_unlabel_new(i,:)-C(j,:));
%     end
%     [~,index] = min(dis);
%     label(i) = trans(index);
% end
label = zeros(size(X_label_new,1),1);

for i = 1:size(X_label_new,1)
    dis = zeros(cluster,1);
    for j = 1:cluster
        dis(j) = norm(X_label(i,:)-C(j,:));
    end
    [~,index] = min(dis);
    label(i) = trans(index);
end
mean(label == Y)