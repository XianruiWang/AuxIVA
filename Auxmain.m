clear; clc;
param.nsou=2;
param.nfft=2048;
param.eps=1e-5;
param.epoch = 100;
[data,fs]=audioread('chamb_mw.wav');
inputSignal = data(1:30*fs,:).';
[demixedSig,W] = AuxIVA(inputSignal,param);