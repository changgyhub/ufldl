function [cost, grad] = orthonormalICACost(theta, visibleSize, numFeatures, patches, epsilon)
%orthonormalICACost - compute the cost and gradients for orthonormal ICA
%                     (i.e. compute the cost ||Wx||_1 and its gradient)

    weightMatrix = reshape(theta, numFeatures, visibleSize);
    
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE -------------------- 
    num_samples = size(patches,2);
    cost = sum(sum((weightMatrix'*weightMatrix*patches-patches).^2))./num_samples+...
            sum(sum(sqrt((weightMatrix*patches).^2+epsilon)));
    grad = (2*weightMatrix*(weightMatrix'*weightMatrix*patches-patches)*patches'+...
        2*weightMatrix*patches*(weightMatrix'*weightMatrix*patches-patches)')./num_samples+...
        (weightMatrix*patches./sqrt((weightMatrix*patches).^2+epsilon))*patches';
    grad = grad(:);

end

