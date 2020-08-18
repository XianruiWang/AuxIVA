function [whiteningMatrix,processedSignal] = preProcess(inputSignal,pcaNums)
%% Centering dataset and whitening it
%   Usage: [meanValue,whiteningMatrix,dewhiteningMatrix,processedSignal] = preProcess(inputSignal)
%   meanValue: mean of input signal
%   whiteningMatrix: matrix transform original signal into uncorrelated signal
%   dewhiteningMatrix: the inverse matrix of whiteningMatrix
%   processedSignal: signal centered and whited
%   inputSignal; original input signal to be processed, which is N*M matrix,
%   N is the number of observations and M is the length of one observation 

%   pcaNums; how many components will be reserved after PCA processing

%   Centering: centeredSignal = inputSignal-mean(inputSignal)
%   Whitening: processedSignal = ED^(-1/2)E'inputSignal where correlation
%   PCA: discard unimportant component which can be seen as noise reduction and data compression
%   matrix Rx=EDE'
%   Author:
%           Xianrui Wang, Center of Intelligent Acoustics and Immersive
%           Communications.
%
%   Contact:
%           wangxianrui@mail.nwpu.edu.cn
%--------------------------------------------------------------------------

% Centering
meanValue = mean(inputSignal,2);
centeredSignal = inputSignal-meanValue*ones(1,size(inputSignal,2));

% PCA
Rx = (centeredSignal*centeredSignal')./size(inputSignal,2);
[E,D] = eig(Rx);
[D_valuesort,index] = sort(diag(D),'descend');
E_sort = E(:,index);
pca_E(:,1:pcaNums) = E_sort(:,1:pcaNums);
pca_D(:,1:pcaNums) =diag(real(D_valuesort(1:pcaNums)));

% Whitening
whiteningMatrix = sqrt(inv(pca_D))*pca_E';
processedSignal = whiteningMatrix*centeredSignal;
fprintf('Data preprocessing accomplished \n');