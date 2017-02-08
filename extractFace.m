%% extract faces from 

load ./train_set/train_raw_img.mat % train_img
load ./train_set_unlabeled/train_unlabeled_raw_img.mat % train_unlabeled_img


desired_face_size = [50 50];

train_face.face_detected = zeros(size(train_img,1),1);
train_face.face = zeros(size(train_img,1),7500,'uint8');


%% labeled data
for i = 1:size(train_img,1)
    disp(['image ' num2str(i)]);
    raw_img = reshape_img(train_img(i,:));
    raw_img = imresize(raw_img,3);
    faceDetector_front = vision.CascadeObjectDetector();
    bbox_f = step(faceDetector_front, raw_img);
    faceDetector_profile = vision.CascadeObjectDetector('ProfileFace');
    bbox_p = step(faceDetector_profile, raw_img);
    
    if isempty(bbox_f)
        bbox = [bbox_f;bbox_p];
    else
        bbox = bbox_f;
    end
    
    num_face = size(bbox,1);     % number of detected faces
    
    face_size = bbox(:,3).*bbox(:,4);
    [~,max_face_ind] = max(face_size);
    
%     videoOut = insertObjectAnnotation(raw_img,'rectangle',bbox,'Face');
%     figure, imshow(videoOut), title('Detected face');
    
    if num_face > 0
        train_face.face_detected(i) = 1;
        face = imcrop(raw_img,bbox(max_face_ind,:));
        face = imresize(face,desired_face_size);
        train_face.face(i,:) = face(:);
        
%         if num_face > 1
%             face2 = imcrop(raw_img,bbox(face_size_rank(end-1),:));
%             face2 = imresize(face2,desired_face_size);
%         end
        
    end
    
%     t = 0;
%     close;
end


% unlabeled data
% for i = 1:size(train_unlebeled_img,1)
%     raw_img = reshape_img(train_img(i,:));
%     faceDetector_front = vision.CascadeObjectDetector();
%     bbox_f = step(faceDetector_front, raw_img);
% end