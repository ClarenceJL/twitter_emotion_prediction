function [ prelabel ] = preval2prelabel( preval )
% Transform predicted value for test data to 1-0 label
% estval: a matrix containing real estimated values
% estlabel: a matrix with the same size as estval, containing 1-0 labels

prelabel = sign(preval);

% handle special case: preval(i) = prelabel(i) = 0
zero_ind = (prelabel == 0);
right_ind = ~zero_ind;
randlabel = randsrc(size(preval,1),size(preval,2));  % "randsrc" generates a "-1" or "1" with equal probability
prelabel = prelabel.*right_ind + randlabel.*zero_ind;

% from +1/-1 to 1/0
prelabel = (prelabel + 1) / 2;

end

