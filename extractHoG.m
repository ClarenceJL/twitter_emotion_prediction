% extract hog feature from raw_images

load ./train_set/train_raw_img.mat % train_img
load ./train_set_unlabeled/train_unlabeled_raw_img.mat % train_unlabeled_img


for i = 1:size(train_img,1)
    raw_img = reshape_img(train_img(i,:));
    train_HOG(i,:) = extractHOGFeatures(raw_img);
end

for i = 1:size(train_unlabeled_img,1)
    raw_img = reshape_img(train_unlabeled_img(i,:));
    train_unlabeled_HOG(i,:) = extractHOGFeatures(raw_img);
end
