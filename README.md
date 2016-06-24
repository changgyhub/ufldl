# UFLDL (Unsupervised Feature Learning and Deep Learning) Tutorial Solutions
Solutions to the Exercises of [UFLDL Tutorial](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial) (2016).

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
* `test.m`: The overall procedure, including Initialise constants and parameters, Load data, Implement softmaxCost (using `softmaxCost.m`), Gradient checking (using `computeNumericalGradient.m` in Exercise 1), Learning parameters (using `softmaxTrain.m` which minimizes `softmaxCost.m` by L-BFGS), Testing with test datas. <br>

##Exercise 5: Self-Taught Learning
#####The following files are the core of this exercise:<br>
* `feedForwardAutoencoder.m`: convert the raw image data to hidden unit activations a(2).<br>
* `stlExercise.m`: The overall procedure, including Setting parameters, Load data from the MNIST database (and divided into labled and unlabled data sets), Train the sparse autoencoder with unlabled data set (like Exercise 2), Extract Features from the Supervised Dataset (using `feedForwardAutoencoder.m`, based on the w(1) form the autoencoder), Train the softmax classifier (based on the input from the extracted features), Testing with test datas. <br>

#####Note:
* Briefly speaking, we firstly use sparse autoencoder to train unlabled data and get w(1) and w(2), then use self-taught learning to  obtain a(2) using w(1), finally use Softmax Regression to train labled data (a(2), y) and optimize w(2). Notice that with fine-tuning (introduced in topic 6), we can also optimize w(1) with optimization methods when training labled data. Here we just optimize w(2).
