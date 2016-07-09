function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

%% Unroll softmaxTheta parameter

% Note: About theta:
% theta(1:hiddenSize*numClasses) -> softmaxTheta
% theta(hiddenSize*numClasses+1:end) -> theta(w and b) of each hidden levels

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% Note : input level: level 1
%        hidden levels: level 2 ~ depth + 1, more specifically, it should
%                       be level 2 and 3, level 3 and 4 ... level depth and depth +1,
%                       where level i is the input level of the stacked autoencoder and
%                       level i+1 is the second level to self-teach
%        softmax level: level depth+2


% 1. Feed Forward Autoencoder for the hidden levels (level 2 ~ depth+1)

depth = numel(stack);  
z = cell(depth+1, 1);                   % 'z' of input and hidden levels: 1 - input, 2 ~ depth+1 - hidden
a = cell(depth+1, 1);                   % 'a' of input and hidden levels: 1 - input, 2 ~ depth+1 - hidden
a{1} = data;

for i = 1:depth                         % compute 'z' and 'a' of hidden levels (2 ~ depth+1)
    z{i+1} = stack{i}.w * a{i} + repmat(stack{i}.b, 1, numCases);  
    a{i+1} = sigmoid(z{i+1});  
end  




% 2. Compute J and ¨Œ J for the softmax level (level depth+2)

M = softmaxTheta * a{depth+1};          % 'z' of the softmax level (depth + 2), using softmaxTheta
M = bsxfun(@minus, M, max(M, [],1));   
M = exp(M);
p = bsxfun(@rdivide, M, sum(M));        % posibility matrix

cost = -1/numCases .* sum(groundTruth(:)'*log(p(:))) + lambda/2 *sum(softmaxTheta(:).^2);    %   J(¦È)
softmaxThetaGrad = -1/numCases .* (groundTruth - p) * a{depth+1}' + lambda * softmaxTheta;   % ¨ŒJ(¦È)




% 3. Back Propagation from the last hidden level to the input level
% (depth ~ 1, we minus one here from depth+1 ~ 2, because f(w(i-1), b(i-1);x(i-1)) = a(i),
%  the parameters come from the previous level).

delta = cell(depth+1);          % ¦Ä, the error
delta{depth+1} =  -(softmaxTheta' * (groundTruth-p)) .* a{depth+1} .* (1-a{depth+1});   % ¦Ä of the last hidden level (depth+1)
  
for layer = depth: -1: 2  
    delta{layer} = (stack{layer}.w * delta{layer+1}) .* a{layer} .* (1-a{layer});   % ¦Ä of previous levels (hidden levels)
end  
  
for layer = depth : -1 :1                 % compute ¨Œw and ¨Œb of all previous levels (hidden levels and input level)
    stackgrad{layer}.w = delta{layer+1} * a{layer}' ./ numCases;  
    stackgrad{layer}.b = sum(delta{layer+1}, 2) ./numCases;  
end  


%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
