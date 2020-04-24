%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                                                           %%%%%
%%%%%                BISMILLAH HIRRAHMA NIRRAHEEM               %%%%%
%%%%%                                                           %%%%%
%%%%%               Programmed By: Muzammil Behzad              %%%%%
%%%%%       Center for Machine Vision and Signal Analysis       %%%%%
%%%%%                     University of Oulu                    %%%%%
%%%%%                       Oulu, Finland                       %%%%%
%%%%%                                                           %%%%%
%%%%%   Email: muzammil.behzad@{oulu.fi, ieee.org, gmail.com}   %%%%%
%%%%%                                                           %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;
clc;
tic;

fprintf('\n==============================================> Simulation Started <==============================================\n');
fprintf('Start Time: %s\n', datestr(now,'HH:MM:SS.FFF\n\n'))

% Load a pre-trained, deep, convolutional network
alex = alexnet;
layers = alex.Layers;

% Loading training images
num_classes = 2;
maskImages = 'mask';
maskData = dir(maskImages);
for i1 = 3:length(maskData)
    filename = [maskImages '\' maskData(i1).name];
    fprintf('Processing %s: %d/%d\n',filename,i1-2,length(maskData)-2);
   	temp_image = im2double(imread(filename));
    Images(:,:,:,i1-2,1) = imresize(temp_image, [layers(1).InputSize(1:2)]);
    Labels(i1-2,1) =  categorical({'With Mask'});
end
nomaskImages = 'no_mask';
nomaskData = dir(nomaskImages);
for i2 = 3:length(nomaskData)
    filename = [nomaskImages '\' nomaskData(i2).name];
    fprintf('Processing %s: %d/%d\n',filename,i2-2,length(nomaskData)-2);
    temp_image = im2double(imread(filename));
    Images(:,:,:,i1-2+i2-2,1) = imresize(temp_image, [layers(1).InputSize(1:2)]);
    Labels(i1-2+i2-2,1) =  categorical({'Without Mask'});
end
 
% Modify the network to use two categories
layers(23) = fullyConnectedLayer(num_classes); 
layers(25) = classificationLayer;

% Set up and split the training data
split_size = 0.80 ;
total_samples = numel(Labels);
idx = randperm(total_samples)  ;
XTrain = Images(:,:,:,idx(1:round(split_size*total_samples)));
YTrain = Labels(idx(1:round(split_size*total_samples))); 
XTest = Images(:,:,:,idx(round(split_size*total_samples)+1:end));
YTest = Labels(idx(round(split_size*total_samples)+1:end));

% Re-train the Network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 10, 'MiniBatchSize', 64);
myNet = trainNetwork(XTrain, YTrain, layers, opts);

% Measure network accuracy
predictedLabels = classify(myNet, XTest); 
accuracy = mean(predictedLabels == YTest)
save('AlexNetMaskDetect','myNet');


toc;
fprintf('End Time: %s\n', datestr(now,'HH:MM:SS.FFF'))
fprintf('\n==============================================> Simulation Ended <==============================================\n');

