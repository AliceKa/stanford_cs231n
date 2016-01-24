# Stanford CS231n: Convolutional Neural Networks for Visual Recognition (Winter 2016)
 
## Assignment 1
This directory contains all the Jupyter notebooks and Python classes used for the assignment. This assignment contains the following algorithms and techniques. The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is used with 50,000 training images and 10,000 test images.labels for each. The test set classification accuracy is quoted after each assignment below. For reference, here is a [CIFAR-10 leaderboard](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).

2. **k-Nearest Neighbour** classification (27.8% accuracy).
3. Multiclass **SVM** classifier (38.0% accuracy).
4. Multiclass **Softmax** classifier (38.0% accuracy).
5. Two-layer **neural network** classifier (53.4% accuracy).
6. **Feature engineering** using colour histograms and Histogram-Of-Gradients (HOGs) (SVM 40.8% accuracy, 60.2% accuracy).
7. **Cross validation** approaches, including a hold-out set, and k-fold cross validation.
8. **Hyperparameter grid searches** for regularization coefficient, learning rate, hidden layer size.

Each assignment is split between Jupyter notebooks in this directory, and supporting classes in the [cs231n/classifiers](cs231n/classifiers) subdirectory. The notebooks contain a similar flow of checking loss and gradient functions using naive implementations. Then a vectorized implementation of the gradient is compared with a numerical gradient. 

A hyperparameter search is used with a cross-validation data set to select the best performance. Then the final test-set accuracy is computed in the final cell.

For more information, see the [course webpage](http://cs231n.github.io/assignments2016/assignment1/). 

### k-Nearest Neighbours
This is a simple algorithm, with a couple of bad requirements:

* The entire training set has to be stored in memory.
* Pairwise distances to be computed between each training example and each test example. You have to find the "nearest" points so a similarity metric is needed.

This isn't a linear classifier, and so doesn't have loss and gradient functions. The algorithm keeps adding the nearest points, and when _k_ points are all from the same training example, it assigns the training example to the test one.

### Linear Classifiers (SVM and Softmax)
Both the SVM and Softmax algorithms are linear classifiers. During training, They take a single training image **xi** as input, and calculate set of output scores using a matrix of weights **W** and a bias term **b**. The bias term is often added as an extra column in the **W** matrix to simplify the calculation.

**y**i = f(**x**i, **W**, **b**) = **W****x**i + **b**


#### Vector/Matrix dimensions:

The shapes of the vectors and matrices above help to understand how the function transforms from an input example to a set of class predictions. The dimensions are listed below. Note that **X** is the entire input image set (N x C), whereas **X**i is a single image at row **i** and of size (1 x C).

 * N : The number of training examples.
 * D : The number of dimensions for each training example.
 * C : The number of classes.

The dimensions of the matrices in the formula are:

|  Matrix  |  Rows  |  Columns  |
|----------|--------|-----------|
| **X**    | N      | D         |
| **W**    | D      | C         |
| **b**    | D      | 1         |
| **y**    | N      | C         |

Once the **y**i is available, this then used to calculate a loss function, and a gradient function. The SVM and Softmax use different loss and gradient functions, as shown below.


#### Loss and Gradient Functions

Both loss functions use the concept of a _correct class_ whose column number matches the labelled value at **y**i. The _incorrect class_ score refers to the scores at columns which aren't equal to **y**i. When calculating the loss and gradient, different calculations are used for each.

The loss function shows how far from an optimal solution the current values in **W** are. It should decrease during training.

The gradient function indicates the direction of greatest slope of **W**, given the current value of **X**. This is used in Stochastic Gradient Descent (SGD) to update the weights, and reduce the loss function.

### Neural Network Classifier
The Neural Network used in the assignment is a 2-layer fully connected net. The size of the hidden dimension is checked during hyperparameter search in the two_layer_net notebook.
