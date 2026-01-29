clear ; close all ; clc ;

%% 画像復元用データセットの読み込み
% 劣化画像（入力）と正解画像（Ground Truth）のペアを含むデータセット
load('cifar10_restoration_dataset.mat');
dataname = 'cifar10';

% 学習・検証・テスト用データ数
numtrn = size(xtrn, 4);
numval = size(xval, 4);
numtst = size(xtst, 4);

% 学習データの可視化
% 劣化画像と正解画像をタイル表示で確認
h10 = figure(10); imshow(imtile(xtrn)); title('Train: Blurred images');
h11 = figure(11); imshow(imtile(ytrn)); title('Train: Correct images');

% 入力画像サイズの設定（CIFAR-10 固定）
% 高さ・幅・チャネル数
imH = 32; 
imW = 32; 
imC = 3;

% ネットワークのチャネル数設定
% basech を変更することでモデルの表現能力を調整可能
basech = 16;
lastch = round(sqrt(imC * basech));

%% skip connection を有する U-Net（3階層構造）
% Encoder–Decoder 構造に skip connection を導入し，
% 空間情報の保持と復元性能の向上を図る
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

    % --- Decoder: レベル3 (4x4 -> 8x8) ---
    transposedConv2dLayer(2, 4*basech, 'Stride',2, 'Name','dec3_upconv')
    reluLayer('Name','dec3_uprelu')
    depthConcatenationLayer(2, 'Name','dec3_concat')  % Encoder レベル3との結合
    convolution2dLayer(3, 4*basech, 'Padding','same', 'Name','dec3_conv1')
    reluLayer('Name','dec3_relu1')
    convolution2dLayer(3, 4*basech, 'Padding','same', 'Name','dec3_conv2')
    reluLayer('Name','dec3_relu2')

    % --- Decoder: レベル2 (8x8 -> 16x16) ---
    transposedConv2dLayer(2, 2*basech, 'Stride',2, 'Name','dec2_upconv')
    reluLayer('Name','dec2_uprelu')
    depthConcatenationLayer(2, 'Name','dec2_concat')  % Encoder レベル2との結合
    convolution2dLayer(3, 2*basech, 'Padding','same', 'Name','dec2_conv1')
    reluLayer('Name','dec2_relu1')
    convolution2dLayer(3, 2*basech, 'Padding','same', 'Name','dec2_conv2')
    reluLayer('Name','dec2_relu2')

    % --- Decoder: レベル1 (16x16 -> 32x32) ---
    transposedConv2dLayer(2, 1*basech, 'Stride',2, 'Name','dec1_upconv')
    reluLayer('Name','dec1_uprelu')
    depthConcatenationLayer(2, 'Name','dec1_concat')  % Encoder レベル1との結合
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

%% skip connection の設定
% Encoder 出力を対応する Decoder に接続
lgraph = connectLayers(lgraph, 'enc1_relu2', 'dec1_concat/in2'); % 32x32
lgraph = connectLayers(lgraph, 'enc2_relu2', 'dec2_concat/in2'); % 16x16
lgraph = connectLayers(lgraph, 'enc3_relu2', 'dec3_concat/in2'); % 8x8

figure(1); plot(lgraph);
analyzeNetwork(lgraph);

%% 学習用ハイパーパラメータ設定
% Adam 最適化手法を使用
options = trainingOptions('adam', ...
    'MaxEpochs',            10, ...
    'MiniBatchSize',        32, ...
    'InitialLearnRate',     1e-3, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.5, ...
    'LearnRateDropPeriod',  5, ...
    'Shuffle',              'every-epoch', ...
    'ValidationData',       {xval, yval}, ...
    'ValidationFrequency',  200, ...
    'Verbose',              true, ...
    'Plots',                'training-progress');

%% 学習
options.ExecutionEnvironment = 'gpu';
options.ValidationData = {xval, yval} ;
options.ValidationFrequency = 200 ;

tic
[net, trninfo] = trainNetwork(xtrn, ytrn, lgraph, options);
toc

%% 推論
ptrn = predict(net, xtrn);
ptst = predict(net, xtst);

%% 定量評価（RMSE・決定係数）
rmsetrn = calc_rmse_imagearray(ytrn, ptrn); % 学習データ
rmsetst = calc_rmse_imagearray(ytst, ptst); % テストデータ
r2trn   = calc_r2_imagearray(ytrn, ptrn);
r2tst   = calc_r2_imagearray(ytst, ptst);

fprintf('\t\tRMSE\t\tR-squared\n');
fprintf('train\t%f\t%f\n', rmsetrn, r2trn);
fprintf('test\t%f\t%f\n', rmsetst, r2tst);

%% 復元結果の可視化（テスト画像）
idx = 1:5;   % 表示する画像インデックス

figure;
set(gcf,'Position',[100 100 1400 900]);

for k = 1:5
    subplot(5,3,3*k-2);
    imshow(ytst(:,:,:,idx(k))); title('Original'); axis off

    subplot(5,3,3*k-1);
    imshow(xtst(:,:,:,idx(k))); title('Degraded'); axis off

    subplot(5,3,3*k);
    imshow(ptst(:,:,:,idx(k))); title('Restored'); axis off
end

sgtitle('Image Restoration Results (Test Images)');

%% 高解像度画像として保存
exportgraphics(gcf, ...
    sprintf('%s_%s_Original_Degraded_Restored.png', mfilename, dataname), ...
    'Resolution',300);

%% 学習済みモデルの保存
outdir = 'training_results';
if ~exist(outdir,'dir')
    mkdir(outdir);
end

% 評価指標を含めたファイル名で保存
modelname = sprintf('%s_%s_RMSEtrn%.4f_RMSEtst%.4f_R2trn%.4f_R2tst%.4f.mat', ...
    mfilename, dataname, rmsetrn, rmsetst, r2trn, r2tst);

save(fullfile(outdir, modelname), 'net');

%% 終了
