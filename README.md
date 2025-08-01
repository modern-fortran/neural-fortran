# neural-fortran

A parallel framework for deep learning.
Read the paper [here](https://arxiv.org/abs/1902.06714).

* [Features](https://github.com/modern-fortran/neural-fortran#features)
* [Getting started](https://github.com/modern-fortran/neural-fortran#getting-started)
  - [Building with fpm](https://github.com/modern-fortran/neural-fortran#building-with-fpm)
  - [Building with CMake](https://github.com/modern-fortran/neural-fortran#building-with-cmake)
* [Examples](https://github.com/modern-fortran/neural-fortran#examples)
* [API documentation](https://github.com/modern-fortran/neural-fortran#api-documentation)
* [Contributing](https://github.com/modern-fortran/neural-fortran#contributing)
* [Acknowledgement](https://github.com/modern-fortran/neural-fortran#acknowledgement)
* [Related projects](https://github.com/modern-fortran/neural-fortran#related-projects)

## Features

* Training and inference of dense (fully connected), convolutional (1-d and 2-d),
  and transformer neural networks
* Stochastic gradient descent optimizers: Classic, momentum, Nesterov momentum,
  RMSProp, Adagrad, Adam, AdamW
* More than a dozen activation functions and their derivatives
* Loss functions and metrics: Quadratic, Mean Squared Error, Pearson Correlation etc.
* Data-based parallelism
* Loading dense and convolutional models from Keras HDF5 (.h5) files
(see the [nf-keras-hdf5](https://github.com/neural-fortran/nf-keras-hdf5) add-on)

### Available layers

| Layer type | Constructor name | Supported input layers | Rank of output array | Forward pass | Backward pass |
|------------|------------------|------------------------|----------------------|--------------|---------------|
| Input | `input` | n/a | 1, 2, 3 | n/a | n/a |
| Embedding | `embedding` | n/a | 2 | ✅ | ✅ |
| Dense (fully-connected) | `dense` | `input1d`, `dense`, `dropout`, `flatten` | 1 | ✅ | ✅ |
| Dropout | `dropout` | `dense`, `flatten`, `input1d` | 1 | ✅ | ✅ |
| Locally connected (2-d) | `locally_connected` | `input`, `locally_connected`, `conv`, `maxpool`, `reshape` | 2 | ✅ | ✅ |
| Convolutional (1-d and 2-d) | `conv` | `input`, `conv`, `maxpool`, `reshape` | 2, 3 | ✅ | ✅ |
| Max-pooling (1-d and 2-d) | `maxpool` | `input`, `conv`, `maxpool`, `reshape` | 2, 3 | ✅ | ✅ |
| Linear (2-d) | `linear2d` | `input2d`, `layernorm`, `linear2d`, `self_attention` | 2 | ✅ | ✅ |
| Self-attention | `self_attention` | `input2d`, `layernorm`, `linear2d`, `self_attention` | 2 | ✅ | ✅ |
| Layer Normalization | `layernorm` | `linear2d`, `self_attention` | 2 | ✅ | ✅ |
| Flatten | `flatten` | `input2d`, `input3d`, `conv1d`, `conv2d`, `maxpool1d`, `maxpool2d`, `reshape` | 1 | ✅ | ✅ |
| Reshape (1-d to 2-d or 3-d) | `reshape` | `dense`, `dropout`, `flatten`, `input1d` | 2, 3 | ✅ | ✅ |

## Getting started

Get the code:

```
git clone https://github.com/modern-fortran/neural-fortran
cd neural-fortran
```

### Dependencies

Required dependencies are:

* A Fortran compiler
* [fpm](https://github.com/fortran-lang/fpm) or
  [CMake](https://cmake.org) to build the code

Optional dependencies are:

* OpenCoarrays (for parallel execution with GFortran)
* BLAS, MKL, or similar (for offloading `matmul` and `dot_product` calls)
* curl (for downloading testing and example datasets)

Compilers tested include:

* flang-new 20.0.0
* gfortran 13.2.0, 14.0.1
* ifort 2021.13.1
* ifx 2024.2.1

### Building with fpm

#### Building in serial mode

With gfortran, the following will create an optimized build of neural-fortran:

```
fpm build --profile release
```

#### Building in parallel mode

If you use GFortran and want to run neural-fortran in parallel,
you must first install [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays).
Once installed, use the compiler wrappers `caf` and `cafrun` to build and execute
in parallel, respectively:

```
fpm build --compiler caf --profile release --flag "-cpp -DPARALLEL"
```

#### Testing with fpm

```
fpm test --profile release
```

For the time being, you need to specify the same compiler flags to `fpm test`
as you did in `fpm build` so that fpm knows it should use the same build
profile.

See the [Fortran Package Manager](https://github.com/fortran-lang/fpm) for more info on fpm.

### Building with CMake

#### Building in serial mode

```
mkdir build
cd build
cmake ..
make
```

Tests and examples will be built in the `bin/` directory.

#### Building in parallel mode

If you use GFortran and want to run neural-fortran in parallel,
you must first install [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays).
Once installed, use the compiler wrappers `caf` and `cafrun` to build and execute
in parallel, respectively:


```
FC=caf cmake .. -DPARALLEL
make
cafrun -n 4 bin/mnist # run MNIST example on 4 cores
```

#### Building with a different compiler

If you want to build with a different compiler, such as Intel Fortran,
specify `FC` when issuing `cmake`:

```
FC=ifort cmake ..
```

for a parallel build of neural-fortran, or

```
FC=ifort cmake ..
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

## Using neural-fortran in your project

You can use the CMake module available [here](cmake/Findneural-fortran.cmake) to
find or fetch an installation of this project while configuring your project. This
module makes sure that the `neural-fortran::neural-fortran` target is always generated regardless
of how the neural-fortran is included in the project.

First, either copy `Findneural-fortran.cmake` to, say, your project's `cmake` directory
and then include it in your `CMakeLists.txt` file:

```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
```
or use the `CMAKE_MODULE_PATH` variable to point to the directory where it is installed.

Next you need to set `neural-fortran_ROOT_DIR` to the directory where neural-fortran is installed
such that `neural-fortran_ROOT_DIR/lib/libneural-fortran.a` exists.

The following should be added in the CMake file of your directory:

```cmake
if(NOT TARGET neural-fortran::neural-fortran)
  find_package(neural-fortran REQUIRED)
endif()
```

and then to use the target in your project:

```cmake
target_link_libraries(your_target PRIVATE neural-fortran::neural-fortran)
```

## Examples

The easiest way to get a sense of how to use neural-fortran is to look at
examples, in increasing level of complexity:

1. [simple](example/simple.f90): Approximating a simple, constant data
  relationship
2. [sine](example/sine.f90): Approximating a sine function
3. [dense_mnist](example/dense_mnist.f90): Hand-written digit recognition
  (MNIST dataset) using a dense (fully-connected) network
4. [cnn_mnist](example/cnn_mnist.f90): Training a CNN on the MNIST dataset
5. [get_set_network_params](example/get_set_network_params.f90): Getting and
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

## Contributing

This [Contributing guide](CONTRIBUTING.md) briefly describes the code organization.
It may be useful to read if you want to contribute a new feature to neural-fortran.

## Acknowledgement

Thanks to all open-source contributors to neural-fortran:
[awvwgk](https://github.com/awvwgk),
[certik](https://github.com/certik),
[ggoyman](https://github.com/ggoyman),
[ivan-pi](https://github.com/ivan-pi),
[jacobwilliams](https://github.com/jacobwilliams),
[jvdp1](https://github.com/jvdp1),
[jvo203](https://github.com/jvo203),
[mathomp4](https://github.com/mathomp4),
[milancurcic](https://github.com/milancurcic),
[OneAdder](https://github.com/OneAdder),
[pirpyn](https://github.com/pirpyn),
[rico07](https://github.com/ricor07),
[rouson](https://github.com/rouson),
[rweed](https://github.com/rweed),
[Spnetic-5](https://github.com/Spnetic-5),
and [scivision](https://github.com/scivision).

Development of convolutional networks and Keras HDF5 adapters in
neural-fortran was funded by a contract from NASA Goddard Space Flight Center
to the University of Miami.
Development of optimizers is supported by the Google Summer of Code 2023 project
awarded to [Fortran-lang](https://github.com/fortran-lang).

<img src="assets/nasa.png" alt="NASA logo">
<img src="assets/gsoc.png" alt="GSoC logo">

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
* [Inference Engine](https://github.com/BerkeleyLab/inference-engine) developed
at the Berkeley Lab by the Computer Languages and Systems Software (CLaSS)
group.

## Impact

Neural-fortran has been used successfully in over a dozen published studies.
See all papers that cite it
[here](https://scholar.google.com/scholar?cites=7315840714744905948).
