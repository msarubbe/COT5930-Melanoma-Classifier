%% Loading Model and Setting up Layers

%Load the Resnet 18 Layers
lgraph = resnet18('Weights','none');

%Set up training datastore
trainingFolder = 'archive/melanoma_cancer_dataset/train';
categories = {'benign', 'malignant'};
trainingDatastore = imageDatastore(fullfile(trainingFolder, categories), 'LabelSource', 'foldernames');
%Using only 40% of the data, to speed up training/testing
[trainingDatastore, unused_train] = splitEachLabel(trainingDatastore, 0.4, 'randomized');
tbl_train = countEachLabel(trainingDatastore);
disp (tbl_train)

%Set up testing datastore
testingFolder = 'archive/melanoma_cancer_dataset/test';
categories = {'benign', 'malignant'};
testingDatastore = imageDatastore(fullfile(testingFolder, categories), 'LabelSource', 'foldernames');
%Using only 40% of the data, to speed up training/testing
[testingDatastore, unused_test] = splitEachLabel(testingDatastore, 0.4, 'randomized');
tbl_test = countEachLabel(testingDatastore);
disp (tbl_test)

%Randomizing unused data to test on later 
random_unused_test = shuffle(unused_test);

%Whenever an image is read from the Datastore, it will be preprocessed
%Using a function in the appendix of this code. 
trainingDatastore.ReadFcn = @(filename)image_preprocess(filename);
testingDatastore.ReadFcn = @(filename)image_preprocess(filename);
unused_test.ReadFcn = @(filename)image_preprocess(filename);
unused_train.ReadFcn = @(filename)image_preprocess(filename);
random_unused_test.ReadFcn = @(filename)image_preprocess(filename);

%Using 4 epochs to speed up training
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testingDatastore, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%Finds the classisfication output layer needed to be replaced
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

%2 classes for classification, benign and malignant
numClasses = 2; 

%Establishes a new learnable layer, based on the type of classification
%layer found earlier
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',20, ...
        'BiasLearnRateFactor',20);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',20, ...
        'BiasLearnRateFactor',20);
end

%Replaces the old layers with the new one 
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
%% Resnet 18 Classifier Training 

resnet_18_classifier = trainNetwork(trainingDatastore,lgraph,options);
%% Resnet 18 Clasifier Testing on Unseen Data

YPred = classify(resnet_18_classifier,testingDatastore);

YValidation = testingDatastore.Labels;
accuracy_unseen = mean(YPred == YValidation);
fprintf("The validation accuracy is: %.2f %%\n", accuracy_unseen * 100);
%% Reesnet 18 Classifier Testing on More Unseen Data
[YPred,scores] = classify(resnet_18_classifier,random_unused_test);

YValidation = random_unused_test.Labels;
accuracy_unseen_random = mean(YPred == YValidation);
fprintf("The validation accuracy on unseen test data: %.2f %%\n", accuracy_unseen_random * 100);



%% Appendix 

function new_image = image_preprocess(filename)
skin_data = imread(filename);
new_image = imresize(skin_data,[224 224]);
end
%Function from: https://www.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html

% findLayersToReplace(lgraph) finds the single classification layer and the
% preceding learnable (fully connected or convolutional) layer of the layer
% graph lgraph.

function [learnableLayer,classLayer] = findLayersToReplace(lgraph)

if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Argument must be a LayerGraph object.')
end

% Get source, destination, and layer names.
src = string(lgraph.Connections.Source);
dst = string(lgraph.Connections.Destination);
layerNames = string({lgraph.Layers.Name}');

% Find the classification layer. The layer graph must have a single
% classification layer.
isClassificationLayer = arrayfun(@(l) ...
    (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|isa(l,'nnet.layer.ClassificationLayer')), ...
    lgraph.Layers);

if sum(isClassificationLayer) ~= 1
    error('Layer graph must have a single classification layer.')
end
classLayer = lgraph.Layers(isClassificationLayer);


% Traverse the layer graph in reverse starting from the classification
% layer. If the network branches, throw an error.
currentLayerIdx = find(isClassificationLayer);
while true
    
    if numel(currentLayerIdx) ~= 1
        error('Layer graph must have a single learnable layer preceding the classification layer.')
    end
    
    currentLayerType = class(lgraph.Layers(currentLayerIdx));
    isLearnableLayer = ismember(currentLayerType, ...
        ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);
    
    if isLearnableLayer
        learnableLayer =  lgraph.Layers(currentLayerIdx);
        return
    end
    
    currentDstIdx = find(layerNames(currentLayerIdx) == dst);
    currentLayerIdx = find(src(currentDstIdx) == layerNames);
    
end



end
