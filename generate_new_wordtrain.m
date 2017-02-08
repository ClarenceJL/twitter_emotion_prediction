function [X_new] = generate_new_wordtrain(topwords_idx, X)
% topword_idx = readtable('../stemwords2.csv');
% topwords_idx = topword_idx.Var1;
X_new = sparse(4500, max(topwords_idx));
for i = 1 : 10000
    X_new(:,topwords_idx(i)) = X(:,i) + X_new(:,topwords_idx(i));
end