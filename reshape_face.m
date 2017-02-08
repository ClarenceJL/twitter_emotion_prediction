%% Description: Reshape 1x7500image into 50x50x3 image
% Input: 1x7500
% Output: 50x50x3 uint8 matrix

function output_img = reshape_face(input_img)
    output_img = zeros(50, 50, 3);
    output_img(:,:,1) = reshape(input_img(1:2500), [50, 50]);
    output_img(:,:,2) = reshape(input_img(2501:5000), [50, 50]);
    output_img(:,:,3) = reshape(input_img(5001:7500), [50, 50]);
    output_img = uint8(output_img);
end