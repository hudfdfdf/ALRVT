function out = getScaleSubwindow(im, pos, base_target_sz, scale_factors, scale_window, scale_model_sz, hog_scale_cell_size)

% code from DSST

num_scales = length(scale_factors);

for s = 1:num_scales
    patch_sz = floor(base_target_sz * scale_factors(s));
    %make sure the size is not to small
    patch_sz = max(patch_sz, 2);
 
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    
    %check for out-of-bounds coordinates, and set them to the values at
    %the borders
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    %extract image
    im_patch = im(ys, xs, :);
    
    % resize image to model size
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), hog_scale_cell_size);
    temp = temp_hog(:,:,1:31);
    
%         case 'fhog'
%         temp_hog = fhog(single(im_patch_resized), hog_scale_cell_size);
%        %w = cf_response_size(2);
%   %     
%      h=size(temp_hog,1);
%      w=size(temp_hog,2); 
%      temp = zeros(h, w, 28, 'single');
%         temp(:,:,2:28) = temp_hog(:,:,1:27);
%         if hog_scale_cell_size > 1
%             im_patch = mexResize(im_patch_resized, [h, w] ,'auto');
%         end
%         % if color image
%         if size(im_patch, 3) > 1
%             im_patch = rgb2gray(im_patch);
%         end
%         temp(:,:,1) = single(im_patch)/255 - 0.5;
    
    
    
    if s == 1
        out = zeros(numel(temp), num_scales, 'single');
    end
    
    % window
    out(:,s) = temp(:) * scale_window(s);
end
