# neural-fortran

[![Build Status](https://travis-ci.org/modern-fortran/neural-fortran.svg?branch=master)](https://travis-ci.org/modern-fortran/neural-fortran)
[![GitHub issues](https://img.shields.io/github/issues/modern-fortran/neural-fortran.svg)](https://github.com/modern-fortran/neural-fortran/issues)

A parallel neural net microframework. 
Read the paper [here](https://arxiv.org/abs/1902.06714).

* [Features](https://github.com/modern-fortran/neural-fortran#features)
* [Getting started](https://github.com/modern-fortran/neural-fortran#getting-started)
  - [Building in serial mode](https://github.com/modern-fortran/neural-fortran#building-in-serial-mode)
  - [Building in parallel mode](https://github.com/modern-fortran/neural-fortran#building-in-parallel-mode)
  - [Building with a different compiler](https://github.com/modern-fortran/neural-fortran#building-with-a-different-compiler)
  - [Building with BLAS or MKL](https://github.com/modern-fortran/neural-fortran#building-with-blas-or-mkl)
  - [Building in double or quad precision](https://github.com/modern-fortran/neural-fortran#building-in-double-or-quad-precision)
  - [Building in debug mode](https://github.com/modern-fortran/neural-fortran#building-in-debug-mode)
* [Examples](https://github.com/modern-fortran/neural-fortran#examples)
  - [Creating a network](https://github.com/modern-fortran/neural-fortran#creating-a-network)
  - [Training the network](https://github.com/modern-fortran/neural-fortran#training-the-network)
  - [Saving and loading from file](https://github.com/modern-fortran/neural-fortran#saving-and-loading-from-file)
  - [MNIST training example](https://github.com/modern-fortran/neural-fortran#mnist-training-example)
 * [Contributing](https://github.com/modern-fortran/neural-fortran#contributing)

## Features

* Dense, fully connected neural networks of arbitrary shape and size
* Backprop with Mean Square Error cost function
* Data-based parallelism
* Several activation functions
* Support for 32, 64, and 128-bit floating point numbers

## Getting started

Get the code:

```
git clone https://github.com/modern-fortran/neural-fortran
```

Dependencies:

* Fortran 2018-compatible compiler
* OpenCoarrays (optional, for parallel execution, gfortran only)
* BLAS, MKL (optional)

### Building in serial mode

```
cd neural-fortran
mkdir build
cd build
cmake .. -DSERIAL=1
make
```

Tests and examples will be built in the `bin/` directory.

### Building in parallel mode

If you use gfortran and want to build neural-fortran in parallel mode,
you must first install [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays).
Once installed, use the compiler wrappers `caf` and `cafrun` to build and execute
in parallel, respectively:

```
FC=caf cmake ..
make
cafrun -n 4 bin/example_mnist # run MNIST example on 4 cores
```

### Building with a different compiler

If you want to build with a different compiler, such as Intel Fortran,
specify `FC` when issuing `cmake`:

```
FC=ifort cmake ..
```

### Building with BLAS or MKL

To use an external BLAS or MKL library for `matmul` calls,
run cmake like this:

```
cmake .. -DBLAS=-lblas
```

where the value of `-DBLAS` should point to the desired BLAS implementation,
which has to be available in the linking path.
This option is currently available only with gfortran.

### Building in double or quad precision

By default, neural-fortran is built in single precision mode
(32-bit floating point numbers). Alternatively, you can configure to build
in 64 or 128-bit floating point mode:

```
cmake .. -DREAL=64
```

or

```
cmake .. -DREAL=128
```

### Building in debug mode

To build with debugging flags enabled, type:

```
cmake .. -DCMAKE_BUILD_TYPE=debug
```

## Examples

### Creating a network

Creating a network with 3 layers (one hidden layer)
with 3, 5, and 2 neurons each:

```fortran
use mod_network, only: network_type
type(network_type) :: net
net = network_type([3, 5, 2])
```

By default, the network will be initialized with the sigmoid activation
function. You can specify a different activation function:

```fortran
net = network_type([3, 5, 2], activation='tanh')
```

or set it after the fact:

```fortran
net = network_type([3, 5, 2])
call net % set_activation('tanh')
```

Available activation function options are: `gaussian`, `relu`, `sigmoid`,
`step`, and `tanh`.
See [mod_activation.f90](https://github.com/modern-fortran/neural-fortran/blob/master/src/lib/mod_activation.f90)
for specifics.

### Training the network

To train the network, pass the training input and output data sample,
and a learning rate, to `net % train()`:

```fortran
program example_simple
  use mod_network, only: network_type
  implicit none
  type(network_type) :: net
  real, allocatable :: input(:), output(:)
  integer :: i
  net = network_type([3, 5, 2])
  input = [0.2, 0.4, 0.6]
  output = [0.123456, 0.246802]
  do i = 1, 500
    call net % train(input, output, eta=1.0)
    print *, 'Iteration: ', i, 'Output:', net % output(input)
  end do
end program example_simple
```

The size of `input` and `output` arrays must match the sizes of the
input and output layers, respectively. The learning rate `eta` determines
how quickly are weights and biases updated.

The output is:

```
 Iteration:            1 Output:  0.470592350      0.764851630    
 Iteration:            2 Output:  0.409876496      0.713752568    
 Iteration:            3 Output:  0.362703383      0.654729187  
 ...
 Iteration:          500 Output:  0.123456128      0.246801868
```

The initial values will vary between runs because we initialize weights
and biases randomly.

### Saving and loading from file

To save a network to a file, do:

```fortran
call net % save('my_net.txt')
```

Loading from file works the same way:

```fortran
call net % load('my_net.txt')
```

### Synchronizing networks in parallel mode

When running in parallel mode, you may need to synchronize the weights
and biases between images. You can do it like this:

```fortran
call net % sync(1)
```

The argument to `net % sync()` refers to the source image from which to
broadcast. It can be any positive number not greater than `num_images()`.

### MNIST training example

The MNIST data is included with the repo and you will have to unpack it first:

```
cd data/mnist
tar xzvf mnist.tar.gz
cd -
```

The complete program:

```fortran
program example_mnist

  ! A training example with the MNIST dataset.
  ! Uses stochastic gradient descent and mini-batch size of 100.
  ! Can be run in serial or parallel mode without modifications.

  use mod_kinds, only: ik, rk
  use mod_mnist, only: label_digits, load_mnist
  use mod_network, only: network_type

  implicit none

  real(rk), allocatable :: tr_images(:,:), tr_labels(:)
  real(rk), allocatable :: te_images(:,:), te_labels(:)
  real(rk), allocatable :: input(:,:), output(:,:)

  type(network_type) :: net

  integer(ik) :: i, n, num_epochs
  integer(ik) :: batch_size, batch_start, batch_end
  real(rk) :: pos

  call load_mnist(tr_images, tr_labels, te_images, te_labels)

  net = network_type([784, 30, 10])

  batch_size = 100
  num_epochs = 10

  if (this_image() == 1) then
    write(*, '(a,f5.2,a)') 'Initial accuracy: ',&
      net % accuracy(te_images, label_digits(te_labels)) * 100, ' %'
  end if

  epochs: do n = 1, num_epochs
    mini_batches: do i = 1, size(tr_labels) / batch_size

      ! pull a random mini-batch from the dataset
      call random_number(pos)
      batch_start = int(pos * (size(tr_labels) - batch_size + 1))
      batch_end = batch_start + batch_size - 1

      ! prepare mini-batch
      input = tr_images(:,batch_start:batch_end)
      output = label_digits(tr_labels(batch_start:batch_end))

      ! train the network on the mini-batch
      call net % train(input, output, eta=3._rk)

    end do mini_batches

    if (this_image() == 1) then
      write(*, '(a,i2,a,f5.2,a)') 'Epoch ', n, ' done, Accuracy: ',&
        net % accuracy(te_images, label_digits(te_labels)) * 100, ' %'
    end if

  end do epochs

end program example_mnist
```

The program will report the accuracy after each epoch:

```
$ ./example_mnist
Initial accuracy: 10.32 %
Epoch  1 done, Accuracy: 91.06 %
Epoch  2 done, Accuracy: 92.35 %
Epoch  3 done, Accuracy: 93.32 %
Epoch  4 done, Accuracy: 93.62 %
Epoch  5 done, Accuracy: 93.97 %
Epoch  6 done, Accuracy: 94.16 %
Epoch  7 done, Accuracy: 94.42 %
Epoch  8 done, Accuracy: 94.55 %
Epoch  9 done, Accuracy: 94.67 %
Epoch 10 done, Accuracy: 94.81 %
```

You can also run this example without any modifications in parallel,
for example on 16 cores using [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays):

```
$ cafrun -n 16 ./example_mnist
```

## Contributing

neural-fortran is currently a proof-of-concept with potential for
use in production. Contributions are welcome, especially for:

* Expanding the network class to other network infrastructures
* Adding other cost functions such as cross-entropy.
* Model-based (`matmul`) parallelism
* Adding more examples
* Others?

You can start at the list of open [issues](https://github.com/modern-fortran/neural-fortran/issues).
