clear ; close all ; clc ;


%% 画像復元用データセットの読み込み
% 劣化画像（入力）と正解画像（Ground Truth）のペアを含むデータ
load('cifar10_restoration_dataset.mat'); dataname = 'cifar10' ;

% 学習・検証・テストデータ数
numtrn = size(xtrn, 4);
numval = size(xval, 4);
numtst = size(xtst, 4);


% 学習データの可視化
% 劣化画像と正解画像をタイル表示
h10 = figure(10); imshow(imtile(xtrn)) ; title('Train: Blurred images') ;
h11 = figure(11); imshow(imtile(ytrn)) ; title('Train: Correct images') ;

% 入力画像サイズの設定（CIFAR-10）
% 高さ・幅・チャネル数
imH = 32; imW = 32; imC = 3;

% ネットワークのチャネル数設定
% basech を変更することでモデル容量を調整可能
basech = 16 ;
lastch = round(sqrt(imC*basech)) ;

%% skip connection を持たない U-Net 風ネットワーク（3階層）
% Encoder–Decoder 構造を持つが、skip connection は使用しない
layers = [
    % 入力層
    imageInputLayer([imH imW imC], 'Name', 'input', 'Normalization', 'zerocenter')


    % --- Encoder: レベル1 (32x32) ---
    convolution2dLayer(3, 1*basech, 'Padding','same', 'Name','enc1_conv1')
    reluLayer('Name','enc1_relu1')
    convolution2dLayer(3, 1*basech, 'Padding','same', 'Name','enc1_conv2')
    reluLayer('Name','enc1_relu2')
    maxPooling2dLayer(2, 'Stride',2, 'Name','enc1_down')  % -> 16x16


    % --- Encoder: レベル2 (16x16) ---
    convolution2dLayer(3, 2*basech, 'Padding','same', 'Name','enc2_conv1')
    reluLayer('Name','enc2_relu1')
    convolution2dLayer(3, 2*basech, 'Padding','same', 'Name','enc2_conv2')
    reluLayer('Name','enc2_relu2')
    maxPooling2dLayer(2, 'Stride',2, 'Name','enc2_down')  % -> 8x8


    % --- Encoder: レベル3 (8x8) ---
    convolution2dLayer(3, 4*basech, 'Padding','same', 'Name','enc3_conv1')
    reluLayer('Name','enc3_relu1')
    convolution2dLayer(3, 4*basech, 'Padding','same', 'Name','enc3_conv2')
    reluLayer('Name','enc3_relu2')
    maxPooling2dLayer(2, 'Stride',2, 'Name','enc3_down')  % -> 4x4


    % --- ボトルネック層（最下層） ---
    convolution2dLayer(3, 8*basech, 'Padding','same', 'Name','btm_conv1')
    reluLayer('Name','btm_relu1')
    convolution2dLayer(3, 8*basech, 'Padding','same', 'Name','btm_conv2')
    reluLayer('Name','btm_relu2')


  % --- Decoder: レベル3 (4x4 -> 8x8) ----
    transposedConv2dLayer(2, 4*basech, 'Stride',2, 'Name','dec3_upconv') % -> 8x8
    reluLayer('Name','dec3_uprelu')
    %depthConcatenationLayer(2, 'Name','dec3_concat') % concat with enc3
    convolution2dLayer(3, 4*basech, 'Padding','same', 'Name','dec3_conv1')
    reluLayer('Name','dec3_relu1')
    convolution2dLayer(3, 4*basech, 'Padding','same', 'Name','dec3_conv2')
    reluLayer('Name','dec3_relu2')


    % --- Decoder: レベル2 (8x8 -> 16x16) ---
    transposedConv2dLayer(2, 2*basech, 'Stride',2, 'Name','dec2_upconv') % -> 16x16
    reluLayer('Name','dec2_uprelu')
    %depthConcatenationLayer(2, 'Name','dec2_concat') % concat with enc2
    convolution2dLayer(3, 2*basech, 'Padding','same', 'Name','dec2_conv1')
    reluLayer('Name','dec2_relu1')
    convolution2dLayer(3, 2*basech, 'Padding','same', 'Name','dec2_conv2')
    reluLayer('Name','dec2_relu2')


    % --- Decoder: レベル1 (16x16 -> 32x32) ---
    transposedConv2dLayer(2, 1*basech, 'Stride',2, 'Name','dec1_upconv') % -> 32x32
    reluLayer('Name','dec1_uprelu')
    %depthConcatenationLayer(2, 'Name','dec1_concat') % concat with enc1
    convolution2dLayer(3, 1*basech, 'Padding','same', 'Name','dec1_conv1')
    reluLayer('Name','dec1_relu1')
    convolution2dLayer(3, 1*basech, 'Padding','same', 'Name','dec1_conv2')
    reluLayer('Name','dec1_relu2')

    % 出力層
    convolution2dLayer(3, lastch, 'Padding','same', 'Name','out_conv1')
    reluLayer('Name','out_relu1')
    convolution2dLayer(3, imC, 'Padding','same', 'Name','out_conv_fin')
    regressionLayer('Name','regressionOutput')
];

lgraph = layerGraph(layers);

% --- Skip connections （本モデルでは未使用）---
% 32x32: enc1_relu2 -> dec1_concat/in2
%lgraph = connectLayers(lgraph, 'enc1_relu2', 'dec1_concat/in2');

% 16x16: enc2_relu2 -> dec2_concat/in2
%lgraph = connectLayers(lgraph, 'enc2_relu2', 'dec2_concat/in2');

% 8x8:  enc3_relu2 -> dec3_concat/in2
%lgraph = connectLayers(lgraph, 'enc3_relu2', 'dec3_concat/in2');

figure(1) ; plot(lgraph) ;
lgraph.Layers
analyzeNetwork(lgraph);

%% 学習用ハイパーパラメータ設定
% Adam 最適化手法を使用
options = trainingOptions('adam', ...
    'MaxEpochs',            10, ...
    'MiniBatchSize',        32, ...
    'InitialLearnRate',     1e-3, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.5, ...
    'LearnRateDropPeriod',  5, ...         % 10ep中で1回減衰
    'Shuffle',              'every-epoch', ...
    'ValidationData',       {xval, yval}, ...
    'ValidationFrequency',  200, ...
    'Verbose',              true, ...
    'Plots',                'training-progress' ...
);

%% 学習
options.ExecutionEnvironment = 'gpu'; 
options.ValidationData = {xval, yval} ;
options.ValidationFrequency = 200 ;
tic
[net, trninfo] = trainNetwork(xtrn, ytrn, lgraph, options) ;
toc

%% 推論
ptrn = predict(net, xtrn) ;
ptst = predict(net, xtst) ;

%% 定量評価（RMSE・決定係数）
rmsetrn = calc_rmse_imagearray(ytrn, ptrn) ; % 学習データのRMSE
rmsetst = calc_rmse_imagearray(ytst, ptst) ; % テストデータのRMSE
r2trn = calc_r2_imagearray(ytrn, ptrn) ; % 学習データの決定係数
r2tst = calc_r2_imagearray(ytst, ptst) ; % テストデータの決定係数

fprintf('\t\tRMSE\t\tR-squared\n') ;
fprintf('train\t%f\t%f\n', rmsetrn, r2trn) ;
fprintf('test\t%f\t%f\n', rmsetst, r2tst) ;

%% ===== 提出用画像作成 =====
idx = 1:5;   % 表示する画像番号（5枚）

figure;
set(gcf,'Position',[100 100 1400 900]);  % 表示サイズを大きく

for k = 1:5
    % 元画像（Ground Truth）
    subplot(5,3,3*k-2);
    imshow(ytst(:,:,:,idx(k)));
    title('Original');
    axis off

    % 劣化画像
    subplot(5,3,3*k-1);
    imshow(xtst(:,:,:,idx(k)));
    title('Degraded');
    axis off

    % 復元画像
    subplot(5,3,3*k);
    imshow(ptst(:,:,:,idx(k)));
    title('Restored');
    axis off
end

sgtitle('Image Restoration Results (Test Images)');

% ===== 300 dpiで保存 =====
exportgraphics(gcf, ...
    sprintf('%s_%s_Original_Degraded_Restored.png', mfilename, dataname), ...
    'Resolution',300);
%% 学習済みモデルの保存

outdir = 'training_results';
if ~exist(outdir,'dir')
    mkdir(outdir);
end

% モデル名をここで定義
modelname = sprintf('%s_%s_RMSEtrn%.4f_RMSEtst%.4f_R2trn%.4f_R2tst%.4f.mat', ...
    mfilename, dataname, rmsetrn, rmsetst, r2trn, r2tst);

% フルパスで保存
save(fullfile(outdir, modelname), 'net');

%% 終了