function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

%0. get w and b for the input level (randomly initialized previously)
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

%1. forward propagation
data_size=size(data);
active_value2=repmat(b1,1,data_size(2));  % repmat£¨A, x, y): repeat matirx x-by-y times
active_value3=repmat(b2,1,data_size(2));
active_value2=sigmoid(W1*data+active_value2);
active_value3=W2*active_value2+active_value3; % identity function

%2. compute error term and cost
ave_square=sum(sum((active_value3-data).^2)./2)/data_size(2);
weight_decay=lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));

p_real=sum(active_value2,2)./data_size(2);
p_para=repmat(sparsityParam,hiddenSize,1);
sparsity=beta.*sum(p_para.*log(p_para./p_real)+(1-p_para).*log((1-p_para)./(1-p_real)));
cost=ave_square+weight_decay+sparsity;

%3. calculate deltas
delta3=(active_value3-data); % here comes the difference
average_sparsity=repmat(sum(active_value2,2)./data_size(2),1,data_size(2));
default_sparsity=repmat(sparsityParam,hiddenSize,data_size(2));
sparsity_penalty=beta.*(-(default_sparsity./average_sparsity)+((1-default_sparsity)./(1-average_sparsity)));
delta2=(W2'*delta3+sparsity_penalty).*((active_value2).*(1-active_value2));

%4. backword propagation
W2grad=delta3*active_value2'./data_size(2)+lambda.*W2;
W1grad=delta2*data'./data_size(2)+lambda.*W1;
b2grad=sum(delta3,2)./data_size(2);
b1grad=sum(delta2,2)./data_size(2);


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
