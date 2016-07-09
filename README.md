# UFLDL Tutorial Solutions
Solutions to the Exercises of [UFLDL (Unsupervised Feature Learning and Deep Learning) Tutorial](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial "UFLDL Tutorial by Andrew Ng, etc.") (2016).
###@I will upload those files when I finish checking Exercise 2 and 10.@

##Exercise 1: Sparse Autoencoder
#####The following files are the core of this exercise:<br>
* `sampleIMAGES.m`: Load IMAGES.mat and randomly choose paterns to train.<br>
* `sparseAutoencoderCost.m`: Front and back propagation. Note that two implementation methods are provided.<br>
* `computeNumericalGradient.m`: Do the gradient test. This part should be skipped for future examples because it cost huge amount of time.<br>
* `test.m`: The overall procedure.<br>

#####Notes:
* We use a vectorized version already in the first implementation of `sparseAutoencoderCost.m`. An unvectored and bit inelegant implementation is commented after it.

##Exercise 2: Vectorized Sparse Autoencoder
#####The following files are the core of this exercise:<br>
* `sparseAutoencoderCost.m`: Front and back propagation. Note that two implementation methods are provided.<br>
* `test.m`: The overall procedure. Notice that this time we use another set of images (and labels), with parameters altered.<br>
* some .m files to read the images (and labels), see `info\Using the MNIST Dataset.docx`.<br>

#####Notes:
* the result of training is still unsatisfying.

##Exercise 3A: PCA 2D
#####The following files are the core of this exercise:<br>
* `pca_2d.m`: Including Finding the PCA basis, Checking xRot, Dimension reduce and replot, PCA Whitening and ZCA Whitening.<br>

##Exercise 3B: PCA and Whitening
#####The following files are the core of this exercise:<br>
* `pca_gen.m`: Including Load and zero mean the data, Implement PCA (and check covariance), Find number of components to retain, PCA with dimension reduction, PCA with whitening and regularization (and check covariance), ZCA whitening.<br>

##Exercise 4: Softmax Regression
#####The following files are the core of this exercise:<br>
* `softmaxCost.m`: Compute the softmax cost function J(θ) and its gradient.<br>
* `softmaxPredict.m`: Compute the predicted lable (classification) using calculated theta and data under test.<br>
* `test.m`: The overall procedure, including Initialise constants and parameters, Load data, Implement softmaxCost (using `softmaxCost.m`), Gradient checking (using `computeNumericalGradient.m` in Exercise 1), Learning parameters (using `softmaxTrain.m` which minimizes `softmaxCost.m` by L-BFGS), Testing with test datas.<br>

##Exercise 5: Self-Taught Learning
#####The following files are the core of this exercise:<br>
* `feedForwardAutoencoder.m`: convert the raw image data to hidden unit activations a(2).<br>
* `stlExercise.m`: The overall procedure, including Setting parameters, Load data from the MNIST database (and divided into labled and unlabled data sets), Train the sparse autoencoder with unlabled data set (like Exercise 2), Extract Features from the Supervised Dataset (using `feedForwardAutoencoder.m`, based on the w(1) form the autoencoder), Train the softmax classifier (based on the input from the extracted features), Testing with test datas.<br>

#####Notes:
* The whole procedure can be explained as:<br>
  1. Use sparse autoencoder to train unlabled data and get w(1) and w(2);<br>
  2. Use self-taught learning to  obtain a(2) using w(1);<br>
  3. Use Softmax Regression to train labled data (a(2), y) and optimize theta (the new w(2) in final network).<br>
* The overall procedure is explained in topic 6.1. Notice that with fine-tuning (introduced in topic 6), we can also optimize w(1) with optimization methods when training labled data.<br><br>

##Exercise 6: Stacked Autoencoder for Digit Classification
######This Exercise is extremely important, you are highly recomanded to read `stackedAECost.m`, `stackedAEPredict.m` and `stackedAEExercise.m` thoroughly.
#####The following files are the core of this exercise:<br>
* `stackedAECost.m`: This function do the fine-tuning, including<br>
  1. Feed Forward Autoencoder for the hidden levels (level 2 ~ depth+1);<br>
  2. Compute J and ▽ J for the softmax level (level depth+2);<br>
  3. Back Propagation from the last hidden level to the input level (depth ~ 1, we minus one here from depth+1 ~ 2, because f(w(i-1), b(i-1);x(i-1)) = a(i), the parameters come from the previous level).<br>

* `stackedAEExercise.m`: The overall procedure, including<br>
  1. Set Parameters, we set depth = 2;<br>
  2. Load data from the MNIST database;<br>
  3. Train the first sparse autoencoder (input level 1, hidden level 2, output level ignored);<br>
  4. Train the second sparse autoencoder (input level 2, hidden level 3 = depth+1, output level ignored);<br>
  5. Train the softmax classifier (input level 3 = depth+1, output level 4 = depth+2);<br>
  6. Finetune softmax model (using `stackedAECost.m`);<br>
  7. Test (using `stackedAEPredict.m`).<br>

* `stackedAEPredict.m`: Use trained network to test data.<br>

#####Notes:
* The levels in `stackedAECost.m` are:<br>
  1. input level: level 1;<br>
  2. hidden levels: level 2 ~ depth+1, more specifically, it should be level 2 and 3, level 3 and 4 ... level depth and depth +1, where level i is the input level of the stacked autoencoder and level i+1 is the second level to self-teach;<br>
  3. softmax level: level depth+2.<br>

##Exercise 7: Linear Decoder on Color Features
#####The following files are the core of this exercise:<br>
* `sparseAutoencoderLinearCost.m`: modified from `sparseAutoencoderCost.m` in Exercise 1, so that f(·) and delta of the last level is set to identity ("linear") to generate color representations rather than 0~1 gray color.<br>
* `linearDecoderExercise.m`: The overall procedure, including Setting parameters, Gradient checking of the linear decoder, Load patches, ZCA whitening, Learning features (using autoencoder with linear decoder), Visualization.<br>

##Exercise 8: Convolution and Pooling
######This Exercise is extremely important, you are highly recomanded to read `cnnExercise.m`, `cnnConvolve.m` and `cnnPool.m` thoroughly.
#####The following files are the core of this exercise:<br>
* `cnnConvolve.m`: This function do the convolution. The return value __convolvedFeatures__ is of dim 4:<br>
  1. numFeatures: equals number of hidden units in the network, we use this as number of features (i.e. convolution kernels/masks/convolution matrices);<br>
  2. numImages: equals number of images to convolve;<br>
  3. imageDim - patchDim + 1: equals the dimension of convoluted image;<br>
  4. imageDim - patchDim + 1: equals the dimension of convoluted image.<br>
  
  Moreover, the 3rd and 4th dimension is composed of __convolvedImage__, which is computed by:<br>

  1. feature: represents the convolution matrix, it is computed by:<br>
    1. obtain optTheta and ZCAWhite: these are the theta and ZCA matrix obtained from the color features from Exercise 7. More specifically, optTheta contains w and b of the neurons, and ZCAWhite represents the processing matrix of ZCA whitening;<br>
    2. we use `w * ZCAWhite` as each feature (convolution matrix), where w is the corresponding weights from the input neurons to the specific hidden neuron, extraced from optTheta.<br>
  
  2. im: represents the patterns of specific image at specific color channel.<br>
  3. `convolvedImage += conv2(im, feature, 'valid')`: the convolution process.<br>

  * As described in topic 8.3, the calculation using w, b and ZCAWhite can be abstracted as:<br>
  “_Taking the preprocessing steps into account, the feature activations that you should compute is σ(W(T(x-u)) + b), where T is the whitening matrix and u is the mean patch. Expanding this, you obtain σ(WTx - WTu + b), which suggests that you should convolve the images with WT rather than W as earlier, and you should add (b - WTu), rather than just b to convolvedFeatures, before finally applying the sigmoid function._”<br>

* `cnnPool.m`: This function do the pooling. The return value pooledFeatures is of dim 4:<br>
  1. numFeatures: number of features (i.e. convolution kernels/masks/convolution matrices);<br>
  2. numImages: equals number of images to convolve;<br>
  3. resultDim: equals floor(convolvedDim / poolDim), which is the result dimension;<br>
  4. resultDim: equals floor(convolvedDim / poolDim), which is the result dimension.<br>
  
  We simply take the mean of each poolDim*poolDim.<br>

* `cnnExercise.m`: The overall procedure, including<br>
  1. Initialization of parameters;<br>
  2. Train a sparse autoencoder (with a linear decoder) to learn: we simply use the result of Exercise 7. Here three objects are used:
    1. optTheta: theta (w and b) of the autoencoder
    2. ZCAWhite: the ZCA whitening matrix
    3. meanPatch: mean of the patches
  
  3. Test convolution: use `cnnConvolve.m` to test the convolution.
  4. Test pooling: use `cnnPool.m` to test the pooling.
  5. Convolve and pool with the dataset: the core part, including<br>
    1. Load train and test sets;<br>
    2. Divide features into groups. (This part can be omitted, it is just for testing and logging. After all we have to convolute through all features/convolution matrices);<br>
    3. convolute and pool train and test datasets.<br>
  
  6. Use pooled features for classification: Here we choose to use softmax classifier;<br>
  7. Test classifier. You should expect to get an accuracy of around 80% on the test images.<br>


  * As described in topic 8.1, step 2 and step 5 (convolution part) can be abstracted as:<br>
  “_Given some large r × c images xlarge, we first train a sparse autoencoder on small a × b patches xsmall sampled from these images, learning k features f = σ(W(1) × xsmall + b(1)) (where σ is the sigmoid function), given by the weights W(1) and biases b(1) from the visible units to the hidden units. For every a × b patch xs in the large image, we compute fs = σ(W(1) × xs + b(1)), giving us fconvolved, a k × (r - a + 1) × (c - b + 1) array of convolved features._”<br>

  * The change of dimension can be abstacted as:<br>
    * autoencode: vector of images to train the autoencoder (whatever×1) ==> convolution matirx (8×8);<br>
    * convolute: pathces (convolution matirx) (8×8) + image to convolute (64×64) ==> convolutedFeature (57×57);<br>
    * pool(size=19): convolutedFeature (57×57) ==> pooledFeature (3×3).<br>

##Exercise 9: Sparse Coding
#####The following files are the core of this exercise:<br>
* `sparseCodingFeatureCost.m`: This function calculates J(s) and ▽J(s) when A is set.<br>
* `sparseCodingWeightCost.m`: This function calculates J(A) and ▽J(A) when s is set. Actually in our process, the optimal solution can be directly derived and therefore this function is useless.<br>
* `sparseCodingExercise.m`: The overall procedure, including<br>
  1. Initialization of parameters. Here each parameter means a lot, and change of anyone will make a huge difference, see Notes;<br>
  2. Load patches sampled from the original image data;<br>
  3. Checking (the checking process is ignored to save time);<br>
  4. Iterative optimization:<br>
    1. Select a random mini-batch;<br>
    2. Initialize s:<br>
      1. Set _s = A^Tx_ (where _x_ is the matrix of patches in the mini-batch);<br>
      2. For each feature in _s_ (i.e. each column of _s_), divide the feature by the norm of the corresponding basis vector in _A_.<br>
    3. Optimize for feature matrix _s_;<br>
    4. Optimize for weight matrix _A_. Actually here we can directly derive the result as discribed before.<br>
    5. Visualize result at the end of this iteration.<br> 

#####Notes in `sparseCodingExercise.m`:
* In Step 0, we choose patches to be 16 × 16 instead of 8 × 8, and number of features to learn to be 18 × 18 instead of 11 × 11 to obtain better visual results. Also, lamda, epsilon and gamma can also be adjusted to obtain better results.<br>
* When iterating, we use 'cg' (conjugate gradient) instead of 'lbfgs', because lbfgs will make steeper steps and lead to worse results. One alternative is to use lbfgs while decreasing iterations (e.g. options.maxIter=15), or increasing dimension of the grouping region for topographic sparse coding (e.g. poolDim=5) so that the sparse code will be used in larger areas and therefore avoid over-accuracy of _s_.<br>

#####Other Notes:
* Since we minimize _s_ and _A_ alternatively, the cost may not be decreasing all the time, but the overall trend should be.<br>
* There are two changes in `sampleIMAGES.m`:
  1. Three parameters (images, patchDim, numPatches) are added to the function, so that we can customize patch choice;<br>
  2. The rescaling from [-1, 1] to [0.1, 0.9] is deleted, because we have to ensure an average of 0. (see comment for explanation).<br>

##Exercise 10: ICA
#####The following files are the core of this exercise:<br>
* `orthonormalICACost.m`: compute J and ▽J. See [here](http://ufldl.stanford.edu/wiki/index.php/Deriving_gradients_using_the_backpropagation_idea "Deriving gradients using the backpropagation idea") for the method of calculating gradient. Notice that here we use L2 norm instead of L1 to compute J. An intoduction to L0, L1, L2 norm can be found at [here](http://blog.csdn.net/zouxy09/article/details/24971995 "机器学习中的范数规则化之（一）L0、L1与L2范数") (in Chinese).<br>
* `ICAExercise.m`: The overall procedure, including Initialization of parameters, Sample patches, ZCA whiten patches, Gradient checking, and Optimization for orthonormal ICA.<br>

#####Notes:
* This exercise will take around 1-2 days for a laptop to run.<br>
