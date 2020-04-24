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

% Script to fine-tune and then train a pre-trained model
% train_AlexNet; % uncomment to re-train the mdoel (the dataset would be needed)

% Script to access machines's webcam for live facial mask detection
LiveMask_AlexNet;

toc;
fprintf('End Time: %s\n', datestr(now,'HH:MM:SS.FFF'))
fprintf('\n==============================================> Simulation Ended <==============================================\n');

