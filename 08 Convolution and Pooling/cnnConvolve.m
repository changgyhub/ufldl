function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

numImages = size(images, 4);
imageDim = size(images, 1);
imageChannels = size(images, 3);

% Instructions:
%   Convolve every feature with every large image here to produce the 
%   numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1) 
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 3 minutes 
%   Convolving with 5000 images should take around an hour
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

% -------------------- YOUR CODE HERE --------------------
% Precompute the matrices that will be used during the convolution. Recall
% that you need to take into account the whitening and mean subtraction
% steps

WT = W * ZCAWhite;                  % see explation on website Step 2a: Implement convolution
b_mean = b - WT * meanPatch;        % see explation on website Step 2a: Implement convolution
% --------------------------------------------------------

convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);

for imageNum = 1:numImages
  for featureNum = 1:numFeatures % equals number of hidden units in the network

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    for channel = 1:imageChannels

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % ---- YOUR CODE HERE ----
      
      patchSize = patchDim * patchDim;
      offset = (channel -1) * patchSize;  
      feature = reshape(WT(featureNum, offset+1 : offset+patchSize), patchDim, patchDim); % feature = zeros(8,8);
      % Note: In Exercise 7, we set visibleSize = patchDim * patchDim * imageChannels, so that
      %       the second dimension of WT can be divided accordingly.
      
      % ------------------------

      % Flip the feature matrix because of the definition of convolution, as explained later
      feature = rot90(squeeze(feature),2);  % fast version of feature = flipud(fliplr(squeeze(feature)));
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----
     
      convolvedoneChannel = conv2(im, feature, 'valid');
      convolvedImage = convolvedImage + convolvedoneChannel;
      
      % ------------------------

    end
    
    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    % ---- YOUR CODE HERE ----

    convolvedImage = sigmoid(convolvedImage + b_mean(featureNum));
    
    % ------------------------
    
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end


end  
  
function sigm = sigmoid(x)  
    sigm = 1./(1+exp(-x));  
end  
