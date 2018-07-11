% GET_FEATURES: Extracting hierachical convolutional features

function feat = get_features(im, cos_window, layers,midx)

global net
global enableGPU

if isempty(net)
%    initial_net_x(layers(1));
       initial_net(37);
end

sz_window = size(cos_window);

% Preprocessing
img = single(im);        % note: [0, 255] range
%vgg19
img = imResample(img, net.normalization.imageSize(1:2));
img= img - net.normalization.averageImage;
%vgg-f
% img = imResample(img, net.meta.normalization.imageSize(1:2));
% 
%  img (:,:,1) = img (:,:,1)- net.meta.normalization.averageImage(1);
%  img (:,:,2) = img (:,:,2)- net.meta.normalization.averageImage(2);
%  img (:,:,3) = img (:,:,3)- net.meta.normalization.averageImage(3);
if enableGPU, img = gpuArray(img); end
dgg=0;
% Run the CNN
%dagnn
if dgg
    net.conserveMemory = 0; 
    net.eval({'data', img}) ;
else
    % 
% simplenn
  res = vl_simplenn(net,img);
end





% Initialize feature maps
%feat = cell(length(layers), 1);
d=0;
for ii = 1:length(layers)
    % Resize to sz_window
    
   if dgg
          if enableGPU
              x =gather(net.vars(layers(ii)+1).value);
          else
              x =(net.vars(layers(ii)+1).value);
          end
   else
         if enableGPU
                 x = gather(res(layers(ii)).x); 
         else
             x = res(layers(ii)).x;       
         end
   end
%     for si=1:512  imshow(mat2gray(x(:,:,si))); pause(0.01);end
 % x=x(:,:,midx);


% 
%     x = imResample(x, sz_window(1:2));
%     
%     % windowing technique
%     if ~isempty(cos_window),
%         x = bsxfun(@times, x, cos_window);
%     end


    
%     
%      xs=sum(x,3);
% %      ap=mean2(xs);
% %      mx=xs.*(xs>ap*1.5);
%    figure(3);
%    imshow((mat2gray(xs)));
%     feat{ii}=x;

    
%     %     for xi=1:512
% %         imshow(mat2gray(x(:,:,xi)));
% %     end
%     
%     figure(3);
% gx=mat2gray(xs);
% indxs=gray2ind(gx);
% % rgbx=ind2rgb(indxs,map);
% imshow(indxs);
% 

% 
% 

 x = imResample(x, sz_window(1:2));
   
dt=size(x,3);
xt(:,:,d+1:d+dt)=x;
d=d+dt;

end
 x=xt(:,:,midx);
%  x = imResample(x, sz_window(1:2));
    % windowing technique
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);
    end
    feat=x;


end
