# neural-fortran

A parallel neural net microframework.
Companion code to Chapter 6 of 
[Modern Fortran: Building Efficient Parallel Applications](https://www.manning.com/books/modern-fortran?a_aid=modernfortran&a_bid=2dc4d442).

## Getting started

### Getting the code

```
git clone https://github.com/modern-fortran/neural-fortran
```

### Dependencies

* Fortran compiler 
* OpenCoarrays (optional, for parallel execution, gfortran only)
* BLAS, MKL (optional)

### Building neural-fortran

```
cd neural-fortran
mkdir build
cd build
cmake ..
make
```

The examples will be built in the `bin/` directory.

#### Building in parallel mode

If you use gfortran and want to build neural-fortran in parallel mode,
you must first install [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays).
Once installed, use the compiler wrappers `caf` and `cafrun` to build and execute
in parallel, respectively:

```
FC=caf cmake ..
make
cafrun -n 4 bin/example_mnist # run MNIST example on 4 cores
```

#### Building in serial mode

If you use gfortran and want to build neural-fortran in serial mode,
configure using the following flag:

```
cmake .. -DSERIAL=1
```

#### Building with a different compiler

If you want to build with a different compiler, such as Intel Fortran, 
specify `FC` when issuing `cmake`:

```
FC=ifort cmake ..
```

#### Building with BLAS or MKL

To use an external BLAS or MKL library for `matmul` calls,
run cmake like this:

```
cmake .. -DBLAS=-lblas
```

where the value of `-DBLAS` should point to the desired BLAS implementation,
which has to be available in the linking path.
This option is currently available only with gfortran.

#### Building with debug flags

To build with debugging flags enabled, type:

```
cmake .. -DCMAKE_BUILD_TYPE=debug
```

### Unpacking the data

If you intend to work with the MNIST dataset, unpack it first:

```
cd data/mnist
tar xzvf mnist.tar.gz
cd -
```

### Examples

TODO

#### Creating a neural net

#### Training

#### Saving and loading from file

#### MNIST training example

## Features

* Dense, fully connected neural networks of arbitrary shape and size
* Backprop with root-mean-square cost function
* Data-based parallelism
* Several activation functions
* MNIST training example
* Support for 32, 64, and 128-bit floating point numbers
