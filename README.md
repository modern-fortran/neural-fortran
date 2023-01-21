# neural-fortran

[![GitHub issues](https://img.shields.io/github/issues/modern-fortran/neural-fortran.svg)](https://github.com/modern-fortran/neural-fortran/issues)

A parallel framework for deep learning.
Read the paper [here](https://arxiv.org/abs/1902.06714).

* [Features](https://github.com/modern-fortran/neural-fortran#features)
* [Getting started](https://github.com/modern-fortran/neural-fortran#getting-started)
  - [Building with fpm](https://github.com/modern-fortran/neural-fortran#building-with-fpm)
  - [Building with CMake](https://github.com/modern-fortran/neural-fortran#building-with-cmake)
* [Examples](https://github.com/modern-fortran/neural-fortran#examples)
* [API documentation](https://github.com/modern-fortran/neural-fortran#api-documentation)
* [Acknowledgement](https://github.com/modern-fortran/neural-fortran#acknowledgement)
* [Related projects](https://github.com/modern-fortran/neural-fortran#related-projects)

## Features

* Training and inference of dense (fully connected) and convolutional neural
  networks
* Loading dense and convolutional models from Keras HDF5 (.h5) files
* Stochastic and mini-batch gradient descent for back-propagation
* Data-based parallelism
* Several activation functions and their derivatives

### Available layers

| Layer type | Constructor name | Supported input layers | Rank of output array | Forward pass | Backward pass |
|------------|------------------|------------------------|----------------------|--------------|---------------|
| Input | `input` | n/a | 1, 3 | n/a | n/a |
| Dense (fully-connected) | `dense` | `input1d`, `flatten` | 1 | ✅ | ✅ |
| Convolutional (2-d) | `conv2d` | `input3d`, `conv2d`, `maxpool2d`, `reshape` | 3 | ✅ | ✅ |
| Max-pooling (2-d) | `maxpool2d` | `input3d`, `conv2d`, `maxpool2d`, `reshape` | 3 | ✅ | ✅ |
| Flatten | `flatten` | `input3d`, `conv2d`, `maxpool2d`, `reshape` | 1 | ✅ | ✅ |
| Reshape (1-d to 3-d) | `reshape` | `input1d`, `dense`, `flatten` | 3 | ✅ | ✅ |

## Getting started

Get the code:

```
git clone https://github.com/modern-fortran/neural-fortran
cd neural-fortran
```

### Dependencies

Required dependencies are:

* A Fortran compiler
* [HDF5](https://www.hdfgroup.org/downloads/hdf5/)
  (must be provided by the OS package manager or your own build from source)
* [functional-fortran](https://github.com/wavebitscientific/functional-fortran),
  [h5fortran](https://github.com/geospace-code/h5fortran),
  [json-fortran](https://github.com/jacobwilliams/json-fortran)
  (all handled by neural-fortran's build systems, no need for a manual install)
* [fpm](https://github.com/fortran-lang/fpm) or
  [CMake](https://cmake.org) for building the code

Optional dependencies are:

* OpenCoarrays (for parallel execution with GFortran)
* BLAS, MKL, or similar (for offloading `matmul` and `dot_product` calls)
* curl (for downloading testing and example datasets)

Compilers tested include:

* gfortran-9.4.0
* ifort-2021.4
* ifx-2021.4

### Building with fpm

#### Building in serial mode

With gfortran, the following will create an optimized build of neural-fortran:

```
fpm build \
  --profile release \
  --flag "-fno-frontend-optimize -I$HDF5INC -L$HDF5LIB"
```

HDF5 is now a required dependency, so you have to provide it to fpm.
The above command assumes that the `HDF5INC` and `HDF5LIB` environment
variables are set to the include and library paths, respectively, of your
HDF5 install.
The `-fno-frontend-optimize` disables some optimizations that may be harmful
when building neural-fortran.

#### Building in parallel mode

If you use GFortran and want to run neural-fortran in parallel,
you must first install [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays).
Once installed, use the compiler wrappers `caf` and `cafrun` to build and execute
in parallel, respectively:

```
fpm build \
  --compiler caf \
  --profile release \
  --flag "-fno-frontend-optimize -I$HDF5INC -L$HDF5LIB"
```

#### Testing with fpm

```
fpm test \
  --profile release \
  --flag "-fno-frontend-optimize -I$HDF5INC -L$HDF5LIB"
```

For the time being, you need to specify the same compiler flags to `fpm test`
as you did in `fpm build` so that fpm knows it should use the same build
profile.

See [Fortran Package Manager](https://github.com/fortran-lang/fpm) for more info on fpm.

### Building with CMake

#### Building in serial mode

```
mkdir build
cd build
cmake .. -DSERIAL=1
make
```

Tests and examples will be built in the `bin/` directory.

#### Building in parallel mode

If you use GFortran and want to run neural-fortran in parallel,
you must first install [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays).
Once installed, use the compiler wrappers `caf` and `cafrun` to build and execute
in parallel, respectively:


```
FC=caf cmake ..
make
cafrun -n 4 bin/mnist # run MNIST example on 4 cores
```

#### Building with a different compiler

If you want to build with a different compiler, such as Intel Fortran,
set the `HDF5_ROOT` environment variable to the root path of your
Intel HDF5 build, and specify `FC` when issuing `cmake`:

```
FC=ifort cmake ..
```

for a parallel build of neural-fortran, or

```
FC=ifort cmake .. -DSERIAL=1
```

for a serial build.

#### Building with BLAS or MKL

To use an external BLAS or MKL library for `matmul` calls,
run cmake like this:

```
cmake .. -DBLAS=-lblas
```

where the value of `-DBLAS` should point to the desired BLAS implementation,
which has to be available in the linking path.
This option is currently available only with gfortran.

#### Building in debug mode

To build with debugging flags enabled, type:

```
cmake .. -DCMAKE_BUILD_TYPE=debug
```

#### Running tests with CMake

Type:

```
ctest
```

to run the tests.

## Examples

The easiest way to get a sense of how to use neural-fortran is to look at
examples, in increasing level of complexity:

1. [simple](example/simple.f90): Approximating a simple, constant data
  relationship
2. [sine](example/sine.f90): Approximating a sine function
3. [dense_mnist](example/dense_mnist.f90): Hand-written digit recognition
  (MNIST dataset) using a dense (fully-connected) network
4. [cnn_mnist](example/cnn_mnist.f90): Training a CNN on the MNIST dataset
5. [dense_from_keras](example/dense_from_keras.f90): Creating a pre-trained
  dense model from a Keras HDF5 file and running the inference.
6. [cnn_from_keras](example/cnn_from_keras.f90): Creating a pre-trained
  convolutional model from a Keras HDF5 file and running the inference.
7. [get_set_network_params](example/get_set_network_params.f90): Getting and
  setting hyperparameters of a network.

The examples also show you the extent of the public API that's meant to be
used in applications, i.e. anything from the `nf` module.

Examples 3-6 rely on [curl](https://curl.se/) to download the needed datasets,
so make sure you have it installed on your system.
Most Linux OSs have it out of the box.
The dataset will be downloaded only the first time you run the example in any
given directory.

If you're using Windows OS or don't have curl for any other reason,
download [mnist.tar.gz](https://github.com/modern-fortran/neural-fortran/files/8498876/mnist.tar.gz)
directly and unpack in the directory in which you will run the example program.

## API documentation

API documentation can be generated with [FORD](https://github.com/Fortran-FOSS-Programmers/ford/).
Assuming you have FORD installed on your system, run

```
ford ford.md
```

from the neural-fortran top-level directory to generate the API documentation in doc/html.
Point your browser to doc/html/index.html to read it.

## Acknowledgement

Thanks to all open-source contributors to neural-fortran:
[@awvwgk](https://github.com/awvwgk),
[@ivan-pi](https://github.com/ivan-pi),
[@jacobwilliams](https://github.com/jacobwilliams),
[@jvdp1](https://github.com/jvdp1),
[@jvo203](https://github.com/jvo203),
[@milancurcic](https://github.com/milancurcic),
[@pirpyn](https://github.com/pirpyn),
[@rouson](https://github.com/rouson),
[@rweed](https://github.com/rweed),
and [@scivision](https://github.com/scivision).

Development of convolutional networks and Keras HDF5 adapters in
neural-fortran was funded by a contract from NASA Goddard Space Flight Center
to the University of Miami.

## Related projects

* [Fortran Keras Bridge (FKB)](https://github.com/scientific-computing/FKB)
by Jordan Ott provides a Python bridge between old (v0.1.0) neural-fortran
style save files and Keras's HDF5 models. As of v0.9.0, neural-fortran
implements the full feature set of FKB in pure Fortran, and in addition
supports training and inference of convolutional networks.
* [rte-rrtmgp-nn](https://github.com/peterukk/rte-rrtmgp-nn) by Peter Ukkonen
is an implementation based on old (v0.1.0) neural-fortran which optimizes for
speed and running on GPUs the memory layout and forward and backward passes of
dense layers.
