function [demixedSig,W] = AuxIVA(mixedSig,param)
%% Independent vector analysis algorithm
%   based on auxiliary-function method
%
%   Usage:
%       [demixedSig,demixMatrix] = AuxIVA(mixedSig,param)
%   Output:
%       demixedSig: demixed signal, desired
%       W: demix matrix for separation
%
%   Input:
%       mixedSig: obeserved signal
%       param:
%           param.nsou: number of sources
%           param.nfft: fft points 
%           param.epoch: epoch nums 
%   Internal parameters
%        win: short time window
%        nOl: number of overlapped points
%
%   Author:
%           Xianrui Wang, Center of Intelligent Acoustics and Immersive
%           Communications.
%
%   Contact:
%           wangxianrui@mail.nwpu.edu.cn
%--------------------------------------------------------------------------
%% STFT 
win = 2*hanning(param.nfft,'periodic')/param.nfft;
nOl = fix(3*param.nfft/4);
[nMic,nTime] = size(mixedSig);
for micIndex = 1:nMic
    X(micIndex,:,:) = stft(mixedSig(micIndex,:)',param.nfft,win,nOl').';
end
nfreq   = size(X,3);
nframe = size(X,2);
%--------------------------------------------------------------------------
%% Memory allocation
Wp  = zeros(param.nsou,param.nsou,nfreq);
Imat = eye(param.nsou);
Q   = zeros(param.nsou,nMic,nfreq);
Xp  = zeros(param.nsou,nframe,nfreq);
S   = zeros(param.nsou,nframe,nfreq);
demixedSig=zeros(param.nsou,length(mixedSig(1,:)));
%--------------------------------------------------------------------------
%% Priciple component analysis
for fIndex=1:nfreq
    [Q(:,:,fIndex),Xp(:,:,fIndex)] = preProcess(X(:,:,fIndex),param.nsou);
    Wp(:,:,fIndex) = eye(param.nsou); 
end
%--------------------------------------------------------------------------
%% Start iterative learning algorithm
%  auxiliary-function with newton approach
for iter = 1:param.epoch
    % demixing 
    Y = squeeze(sum(bsxfun(@times,permute(Xp,[2,3,4,1]),....
        permute(Wp,[4,3,1,2])),4));
    R = sqrt(squeeze(sum(abs(Y).^2,2)));
    Gr = 1 ./(R+param.eps);
    % construct MM in each frequency bin
    for fIndex=1:nfreq
        x = squeeze(Xp(:,:,fIndex));
        for sIndex = 1:param.nsou
            Rx = bsxfun(@times, permute(x,[2,1,3]),permute(conj(x),[2,3,1]));
            Vk = squeeze(sum(bsxfun(@times,Rx,Gr(:,sIndex)),1))/nframe;
            WV = squeeze(Wp(:,:,fIndex))*Vk;
            w  = WV\Imat(:,sIndex);
            w = w/sqrt(w'*Vk*w);
            Wp(sIndex,:,fIndex) = conj(w);
        end
    end
end
%--------------------------------------------------------------------------
%% Correct scaling permutation, minimal distortion priciple
for fIndex = 1:nfreq
    W(:,:,fIndex) = Wp(:,:,fIndex)*Q(:,:,fIndex);
    W(:,:,fIndex) = diag(diag(pinv(W(:,:,fIndex))))*W(:,:,fIndex);
end
%--------------------------------------------------------------------------
%% frequency-wise smooth
W(:,:,1)=(9*W(:,:,1)+W(:,:,2))/10;
for fIndex = 2:nfreq-1
    W(:,:,fIndex) = (W(:,:,fIndex-1)+8*W(:,:,fIndex)+W(:,:,fIndex+1))/10;
end
W(:,:,nfreq)=(9*W(:,:,nfreq)+W(:,:,nfreq-1))/10;
%--------------------------------------------------------------------------
%% separate frequency components
for fIndex=1:nfreq
    S(:,:,fIndex)=W(:,:,fIndex)*X(:,:,fIndex);
end
%--------------------------------------------------------------------------
%% sythethesis demixed signal
for sIndex=1:param.nsou
    demixedSig(sIndex,:)=istft((squeeze(S(sIndex,:,:)).'),nTime,win,nOl)';
end