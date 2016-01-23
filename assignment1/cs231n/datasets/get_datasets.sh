# Get CIFAR10
# OS X doesn't include wget by default, but it does have curl
# wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

curl -O http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
