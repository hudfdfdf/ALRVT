function initial_net(startL)
% INITIAL_NET: Loading VGG-Net-19

global net;
%%simplenn
net= load(fullfile('model', 'imagenet-vgg-verydeep-19.mat'));
%net = load(fullfile('matconvnet-model','imagenet-vgg-m-1024.mat'));
%net = load(fullfile('matconvnet-model', 'imagenet-matconvnet-vgg-f.mat'));


%Remove the fully connected layers and classification layer
% net.layers(19+1:end) = [];
net.layers(startL+1:end) = [];
% for i=1:12
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
% net = dagnn.DagNN.loadobj(load('matconvnet-model/imagenet-resnet-101-dag.mat')) ;
% 
% for i=345:-1:141
% net.removeLayer(net.layers(i).name) ; 
% end
% net.mode ='test';


% Switch to GPU mode
global enableGPU;
if enableGPU
    %simple NN
    net = vl_simplenn_move(net, 'gpu');
%    %dagnn
%      net.move( 'gpu');
end

end