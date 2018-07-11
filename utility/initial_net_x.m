function initial_net_x(startL)
% INITIAL_NET: Loading VGG-Net-19


global net;
global enableGPU;
global boxreg_net;
% for trainning cnn
opts.useGpu = true;

% model def
% test policy
opts.batchSize_test = 256; % <- reduce it in case of out of gpu memory

% bounding box regression
opts.bbreg = true;
opts.bbreg_nSamples = 1000;

% learning policy
opts.batchSize = 128;
opts.batch_pos = 32;
 opts.batch_neg = 96;

% initial training policy
opts.learningRate_init = 0.0001; % x10 for fc6
opts.maxiter_init = 30;

opts.nPos_init = 500;
opts.nNeg_init = 5000;
opts.posThr_init = 0.7;
opts.negThr_init = 0.5;

% update policy
opts.learningRate_update = 0.0003; % x10 for fc6
opts.maxiter_update = 10;

opts.nPos_update = 50;
opts.nNeg_update = 200;
opts.posThr_update = 0.7;
opts.negThr_update = 0.3;

opts.update_interval = 10; % interval for long-term update

% data gathering policy
opts.nFrames_long = 100; % long-term period
opts.nFrames_short = 20; % short-term period

% cropping policy
opts.input_size = 224;
opts.crop_mode = 'wrap';
opts.crop_padding = 16;

% scaling policy
opts.scale_factor = 1.05;

% sampling policy
opts.nSamples = 256;
opts.trans_f = 0.6; % translation std: mean(width,height)*trans_f/2
opts.scale_f = 1; % scaling std: scale_factor^(scale_f/2)

%     opts.imgSize = size(orgim);
% 
%       
%     pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_init*2, opts, 0.1, 5);
% r = overlap_ratio(pos_examples,targetLoc);
% pos_examples = pos_examples(r>opts.posThr_init,:);
% pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);
% 
% neg_examples = [gen_samples('uniform', targetLoc, opts.nNeg_init, opts, 1, 10);...
%     gen_samples('whole', targetLoc, opts.nNeg_init, opts)];
% r = overlap_ratio(neg_examples,targetLoc);
% neg_examples = neg_examples(r<opts.negThr_init,:);
% neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);
% 
% examples = [pos_examples; neg_examples];
% pos_idx = 1:size(pos_examples,1);
% neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
% 
% ims = mdnet_extract_regions(orgim, examples, opts);
% pos_data = ims(:,:,:,pos_idx);
% neg_data = ims(:,:,:,neg_idx);
% if enableGPU
%     pos_data=gpuArray(pos_data);
%     neg_data=gpuArray(neg_data);
% end

%%simplenn
%net= load(fullfile('model', 'imagenet-vgg-verydeep-19.mat'));
%net = load(fullfile('matconvnet-model','imagenet-vgg-m-1024.mat'));

%   
%  switch net.layers{1,i}.type
%      case 'conv'
%       net.layers{1,i}.filters=net.layers{i}.weights{1,1};
%        net.layers{1,i}.biases=net.layers{i}.weights{1,2}';
%       % net.layers{1,i}.weights=[];
%   end
% end

 %net = dagnn.DagNN.loadobj(load('model/imagenet-googlenet-dag.mat')) ;
%dagnn
dgg=0;
if dgg
    net = dagnn.DagNN.loadobj(load('matconvnet-model/fast-rcnn-vgg16-pascal07-dagnn.mat')) ;

%     for i=19:-1:13
%     net.removeLayer(net.layers(i).name) ; 
%     end
    net.mode ='test';
else
    net = load(fullfile('matconvnet-model', 'imagenet-matconvnet-vgg-f.mat'));
%     boxreg_net = load(fullfile('matconvnet-model', 'tracker.mat'));
    %Remove the fully connected layers and classification layer
    % fine_tuning the network for the first frame;

% %% Learning CNN
% fprintf('  training cnn...\n');
for i=1:numel(net.layers)
         switch net.layers{1,i}.type
             case 'conv'
                 net.layers{1,i}.filters=net.layers{i}.weights{1,1};
                 net.layers{1,i}.biases=net.layers{i}.weights{1,2};
                 net.layers{1,i}.filtersLearningRate=0.01;
                 % net.layers{1,i}.weights=[];
         end
end
net.layers{1,19} = struct('type', 'softmaxloss', 'name', 'loss')
net.layers{1,18} = struct('type', 'conv', ...
    'name', 'fc8n', ...
    'filters', 0.01 * randn(1,1,4096,2,'single'), ...
    'biases', zeros(1, 2, 'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'filtersLearningRate', 10, ...
    'weightDecay',[1 0] );
    if enableGPU
    if dgg
        net.move( 'gpu');
    else
         net= vl_simplenn_move(net, 'gpu');
    end

    
end
    
    
opts.net=net;
% net = mdnet_finetune_hnm(net,pos_data,neg_data,opts,...
%     'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);
    
    
    net.layers(startL+1:end) = [];
    
end


% Switch to GPU mode
if enableGPU
    if dgg
        net.move( 'gpu');
    else
        net = vl_simplenn_move(net, 'gpu');
%         boxreg_net = vl_simplenn_move(boxreg_net, 'gpu');
    end

    
end

end
