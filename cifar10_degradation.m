clear; close all; clc;

% --- load raw CIFAR-10 ---
S = load('cifar10_raw.mat');

Itrn_all = im2double(S.data);
Itst     = im2double(S.data_tst);


Itrn_all = im2double(Itrn_all);
Itst     = im2double(Itst);

numtrn = 45000;
numval = 5000;

ytrn = Itrn_all(:,:,:,1:numtrn);
yval = Itrn_all(:,:,:,numtrn+1:numtrn+numval);
ytst = Itst;

%% sanity check
figure;
for k = 1:5
    subplot(5,1,k); imshow(ytrn(:,:,:,k)); title(sprintf('Original #%d',k));
end

%% degradation process

sigma_blur  = 2.0;
sigma_noise = 0.05;
scale = 2;

xtrn = zeros(size(ytrn), 'like', ytrn);
xval = zeros(size(yval), 'like', yval);
xtst = zeros(size(ytst), 'like', ytst);


% train
for i = 1:size(ytrn,4)
    xtrn(:,:,:,i) = degrade_one_image( ...
        ytrn(:,:,:,i), sigma_blur, sigma_noise, scale);
end

% validation
for i = 1:size(yval,4)
    xval(:,:,:,i) = degrade_one_image( ...
        yval(:,:,:,i), sigma_blur, sigma_noise, scale);
end

% test (never used for training)
for i = 1:size(ytst,4)
    xtst(:,:,:,i) = degrade_one_image( ...
        ytst(:,:,:,i), sigma_blur, sigma_noise, scale);
end

%% save dataset for restoration learning (Enc-Dec / U-Net common)

save('cifar10_restoration_dataset.mat', ...
     'xtrn','ytrn', ...
     'xval','yval', ...
     'xtst','ytst', ...
     '-v7.3');

%% visualize check (5 samples)
figure;

for k = 1:5
    clf;
    imshow(ytrn(:,:,:,k));
    title(sprintf('Original #%d', k));
    pause(1);

    clf;
    imshow(xtrn(:,:,:,k));
    title(sprintf('Degraded #%d', k));
    pause(1);
end

disp('=== Degradation finished successfully ===');