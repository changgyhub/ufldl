# UFLDL (Unsupervised Feature Learning and Deep Learning) Tutorial Solutions
Solutions to the Exercises of [UFLDL Tutorial](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial "UFLDL Tutorial by Andrew Ng, etc.") (2016).
##@I will upload those files once I finish all of the exercises.@

##Exercise 1: Sparse Autoencoder
#####The following files are the core of this exercise:<br>
* `sampleIMAGES.m`: Load IMAGES.mat and randomly choose paterns to train<br>
* `sparseAutoencoderCost.m`: Front and back propagation. Note that two implementation methods are provided.<br>
* `computeNumericalGradient.m`: Do the gradient test. This part should be skipped for future examples because it cost huge amount of time.<br>
* `test.m`: The overall procedure<br>

#####Note:
* We use a vectorized version already in the first implementation of `sparseAutoencoderCost.m`. An unvectored and bit inelegant implementation is commented after it.

##Exercise 2: Vectorized Sparse Autoencoder
#####The following files are the core of this exercise:<br>
* `sparseAutoencoderCost.m`: Front and back propagation. Note that two implementation methods are provided.<br>
* `test.m`: The overall procedure. Notice that this time we use another set of images (and labels), with parameters altered.<br>
* some .m files to read the images (and labels), see `info\Using the MNIST Dataset.docx`<br>

#####Note:
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

#####Note:
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

#####Note:
* The levels in `stackedAECost.m` are:<br>
  1. input level: level 1;<br>
  2. hidden levels: level 2 ~ depth+1, more specifically, it should be level 2 and 3, level 3 and 4 ... level depth and depth +1, where level i is the input level of the stacked autoencoder and level i+1 is the second level to self-teach;<br>
  3. softmax level: level depth+2.<br>

##Exercise 7: Linear Decoder on Color Features
#####The following files are the core of this exercise:<br>
* `sparseAutoencoderLinearCost.m`: modified from `sparseAutoencoderCost.m` in Exercise 1, so that f(·) and delta of the last level is set to identity ("linear") to generate color representations rather than 0~1 gray color.<br>
* `linearDecoderExercise.m`: The overall procedure, including Setting parameters, Gradient checking of the linear decoder, Load patches, ZCA whiting, Learning features (using autoencoder with linear decoder), Visualization.<br>

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
    1. obtain optTheta and ZCAWhite: these are the theta and ZCA matrix obtained from the color features from Exercise 7. More specifically, optTheta contains w and b of the neurons, and ZCAWhite represents the processing matrix of ZCA whiting;<br>
    2. we use `w * ZCAWhite` as each feature (convolution matrix), where w is the corresponding weights from the input neurons to the specific hidden neuron, extraced from optTheta.<br>
  
  2. im: represents the patterns of specific image at specific color channel.<br>
  3. `convolvedImage += conv2(im, feature, 'valid')`: the convolution process.<br>

* `cnnPool.m`: This function do the pooling. The return value pooledFeatures is of dim 4:<br>
  1. numFeatures: number of features (i.e. convolution kernels/masks/convolution matrices);<br>
  2. numImages: equals number of images to convolve;<br>
  3. resultDim: equals floor(convolvedDim / poolDim), which is the result dimension;<br>
  4. resultDim: equals floor(convolvedDim / poolDim), which is the result dimension.<br>
  
  We simply take the mean of each poolDim*poolDim.<br>

* `cnnExercise.m`: The overall procedure, including<br>
  1. ;<br>


