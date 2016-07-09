%% CS294A/CS294W Programming Assignment Starter Code
%  
%  Instructions
%  ------------
%  Step 0 and 1 are modified for this exercise.
%  Original step 2 (implementing cost and grad) and 3 (checking) are skipped to save time.
%
%
%%======================================================================
%% STEP 0: Implement MNIST Images and Labels

images = loadMNISTImages('mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
 
% We are using display_network from the autoencoder code
display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));
figure;  % enable matlab to open two different windows

%% STEP 1: Set Relevant parameters
%  These parameters are modified for Exercise 2, to better fit the images

visibleSize = 28*28;  % number of input units 
hiddenSize = 196;     % number of hidden units 
sparsityParam = 0.1;  % desired average activation of the hidden units.
                          % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p", in the lecture notes).
lambda = 3e-3;        % weight decay parameter   
beta = 3;             % weight of sparsity penalty term
patches = images(:,1:10000);

%%======================================================================
%% STEP 2: start training with minFunc (L-BFGS).

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                                   theta, options);

%%======================================================================
%% STEP 3: Visualization 

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file

