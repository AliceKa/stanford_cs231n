# Stanford CS231n: Convolutional Neural Networks for Visual Recognition (Winter 2016)
 
## Introduction
This repo contains the assignments and links from the Winter 2016 offering of Stanford's CS231n course. I really appreciate Stanford's decision to open up this course for the public to follow along. This field is so new there is no reference textbook, only recent papers from 2012 and onwards.

## Links
There are a few different webpages that are used for the course.

* [Course page](http://cs231n.stanford.edu)
* [Syllabus](http://cs231n.stanford.edu/syllabus.html)
* [Course Notes](http://cs231n.github.io)

## Assignments
There are three assignments in the course, and a project.

* [Assignment 1](http://cs231n.github.io/assignments2016/assignment1/): Image classification, kNN, SVM, Softmax, Neural Network. [My solutions](assignment1/README.md).
* [Assignment 2](http://cs231n.github.io/assignments2016/assignment2/): Fully-connected nets, batch normalization, dropout, convolutional nets. 
* [Assignment 3](http://cs231n.github.io/assignments2016/assignment3/): _TBD_.


## Numpy tips and tricks
The course uses Python and Numpy to code up Machine learning algorithms. The course notes include a great (introduction)[http://cs231n.github.io/python-numpy-tutorial/]. This is a list of useful tricks I've used in the course.

```Python
# Initializing the array dW to 0 with the same size as W.
dW = np.zeros_like(W)

# Adding a new axis dimension of 1 (for broadcasting)
y = y[:, np.newaxis]

# Storing a value on each row, according to a column vector 
rows = np.arange(num_train) # Create an array of row indices
rows = rows[:, np.newaxis] # Convert array from (500,) to (500,1)
y # y contains an array of column values to index below
y = y[:, np.newaxis] # Convert array from (500,) to (500,1)
margins[rows, y] = 0

# Creating a column vector from a matrix, with columns for each row in the y vector
correctProbabilities = probabilities[range(num_examples),y])

# Setting all elements matching a condition to a value
margins[margins > 0] = 1
```


