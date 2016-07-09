function patches = sampleIMAGES(images, patchDim, numPatches)
% sampleIMAGES
% Returns 10000 patches for training

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchDim*patchDim, numPatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

%% Implementation:
% randi(m, x, y) : return a x-by-y matrix with random int entries in [1, m]
% reshape(image, x, y): reshape a matrix to x-by-y

tic
image_size=size(images);
i=randi(image_size(1)-patchDim+1,1,numPatches);
j=randi(image_size(2)-patchDim+1,1,numPatches);
k=randi(image_size(3),1,numPatches);
for num=1:numPatches
        patches(:,num)=reshape(images(i(num):i(num)+patchDim-1,j(num):j(num)+patchDim-1,k(num)),1,patchDim*patchDim);
end
toc

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
%

  patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd; % according to 3 sigma method, 95% data are in this range.



% Note: The process below will make the average to 0.4. In Bayesian view, 
%       A in the model shoule be random variables with an average of 0, 
%       so that the product "As" will also have an average of 0. 
%       Therefore these lines should be deleted.


% Rescale from [-1,1] to [0.1,0.9]
% patches = (patches + 1) * 0.4 + 0.1;

end
