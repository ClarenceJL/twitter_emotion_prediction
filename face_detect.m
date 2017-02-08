function face_test = face_detect(img_test)
faceDetector1 = vision.CascadeObjectDetector('MergeThreshold', 1);
faceDetector2 = vision.CascadeObjectDetector('ClassificationModel', ...
    'UpperBody', 'MergeThreshold', 1);

N = size(img_test, 1);
face_test = zeros(N, 7500);

for i = 1:N
    row = img_test(i,:);
    Img = uint8(reshape(row,[100 100 3]));
    
    bboxes1 = step(faceDetector1, Img);
    if size(bboxes1,1)>0
        I_new = imcrop(Img, bboxes1(end,:));
    else
        bboxes2 = step(faceDetector2, Img);
        if size(bboxes2,1)>0
            I_new = imcrop(Img, bboxes2(end,:));
        else
            I_new = Img;
        end
    end
    
    I_new = imresize(I_new, [50 50]);
    I_new_row = reshape(I_new, [7500,1]);  
    face_test(i,:) = I_new_row;
end
end

