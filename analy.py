image = imread("C:\Users\DELCY LARA\Downloads\miniproject\facedet.jpg");
 
% Separate the RGB channels
R = double(image(:,:,1));
G = double(image(:,:,2));
B = double(image(:,:,3));
 
numBins = 50;
 
[countsR, edgesR] = histcounts2(R(:), R(:), 'NumBins', numBins);
[countsG, edgesG] = histcounts2(G(:), G(:), 'NumBins', numBins);
[countsB, edgesB] = histcounts2(B(:), B(:), 'NumBins', numBins);
 
centersR = edgesR(1:end-1) + diff(edgesR)/2;
centersG = edgesG(1:end-1) + diff(edgesG)/2;
centersB = edgesB(1:end-1) + diff(edgesB)/2;
 
figure;
subplot(1,3,1);
surf(centersR, centersR, countsR', 'EdgeColor', 'none', 'FaceColor', 'r');
xlabel('R Channel');
ylabel('R Channel');
zlabel('Frequency');
title('Histogram of R Channel');
subplot(1,3,2);
surf(centersG, centersG, countsG', 'EdgeColor', 'none', 'FaceColor', 'g');
xlabel('G Channel');
ylabel('G Channel');
zlabel('Frequency');
title('Histogram of G Channel');
subplot(1,3,3);
surf(centersB, centersB, countsB', 'EdgeColor', 'none', 'FaceColor', 'b');
xlabel('B Channel');
ylabel('B Channel');
zlabel('Frequency');
title('Histogram of B Channel');
PSNR ANS SSIM
image1 = imread("C:\Users\DELCY LARA\Downloads\miniproject\facedet.jpg");
image2 = imread("C:\Users\DELCY LARA\Downloads\miniproject\encrypt1.jpg");
 
minHeight = min(size(image1, 1), size(image2, 1));
minWidth = min(size(image1, 2), size(image2, 2));
image1 = imresize(image1, [minHeight, minWidth]);
image2 = imresize(image2, [minHeight, minWidth]);
 
getPSNR = @(image1, image2) 10 * log10((255^2) / (sum((double(image1(:)) - double(image2(:))).^2) / numel(image1)));
 
getMSSIM = @(image1, image2) ssim(image1, image2);
 
psnr_ = getPSNR(image1, image2);
 
ssim_ = getMSSIM(image1, image2);
 
fprintf('PSNR = %f\n', psnr_);
fprintf('SSIM = %f\n', ssim_);

NPCR AND UACI

clear
 
ciphertext1=double(imread('ciphertext.png'));
 
ciphertext2=double(imread('ciphertext2.png'));
 
[rows,cols]=size(ciphertext1);
 
NPCR=100sum(sum(ciphertext1~=ciphertext2))/(rowscols)
UACI=100sum(sum(abs(ciphertext1-ciphertext2)))/(rowscols*255)
