clear;clc;
close all;

%% load wav files

[xm(:, 1), ~] = audioread('micCh1.wav');
[xm(:, 2), param.fs] = audioread('micCh2.wav');

%% source separation

param.theta = [150; 30];
param.lambda = [0.5,0.5];
param.c = [0; 1];
param.cs = 340;
param.delta = 5e-2;
param.nsou=2;
param.nfft=2048;
param.eps=1e-5;
param.epoch = 100;

[demixedSig,W] = GCAV_IVA(xm.',param);



