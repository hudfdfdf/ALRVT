% GET_FEATURES: Extracting hierachical convolutional features

function feat = get_featuresMu(im, cos_window, layers,midx,modeltype)

global net
global enableGPU
if isempty(net)
    if modeltype==1
        initial_net(layers(1));
    else
        initial_net_x(layers(1));
    end
        
end


sz_window = size(cos_window);

% Preprocessing
img = single(im);        % note: [0, 255] range
if modeltype==1
        img = imResample(img, net.normalization.imageSize(1:2));
        img= img - net.normalization.averageImage;
else
     img = imResample(img, net.meta.normalization.imageSize(1:2));
      img (:,:,1) = img (:,:,1)- net.meta.normalization.averageImage(1);
      img (:,:,2) = img (:,:,2)- net.meta.normalization.averageImage(2);
      img (:,:,3) = img (:,:,3)- net.meta.normalization.averageImage(3);
end

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
feat = cell(length(layers), 1);

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
    x=x(:,:,midx{ii});

    x = imResample(x, sz_window(1:2));
    
    % windowing technique
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);
    end
    
    feat{ii}=x;
end

end
