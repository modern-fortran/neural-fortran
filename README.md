# neural-fortran

[![GitHub issues](https://img.shields.io/github/issues/modern-fortran/neural-fortran.svg)](https://github.com/modern-fortran/neural-fortran/issues)

A parallel neural net microframework. 
Read the paper [here](https://arxiv.org/abs/1902.06714).

* [Features](https://github.com/modern-fortran/neural-fortran#features)
* [Getting started](https://github.com/modern-fortran/neural-fortran#getting-started)
  - [Building with fpm](https://github.com/modern-fortran/neural-fortran#building-with-fpm)
  - [Building with CMake](https://github.com/modern-fortran/neural-fortran#building-with-cmake)
* [Examples](https://github.com/modern-fortran/neural-fortran#examples)
* [API documentation](https://github.com/modern-fortran/neural-fortran#api-documentation)
* [Contributors](https://github.com/modern-fortran/neural-fortran#contributors)
* [Related projects](https://github.com/modern-fortran/neural-fortran#related-projects)

## Features

* Dense, fully connected neural layers
* Convolutional and max-pooling layers (experimental, forward propagation only)
* Stochastic and mini-batch gradient descent for back-propagation
* Data-based parallelism
* Several activation functions

### Available layer types

| Layer type | Constructor name | Supported input layers | Rank of output array | Forward pass | Backward pass |
|------------|------------------|------------------------|----------------------|--------------|---------------|
| Input (1-d and 3-d) | `input` | n/a | 1, 3 | n/a | n/a |
| Dense (fully-connected) | `dense` | `input` (1-d) | 1 | ✅ | ✅ |
| Convolutional (2-d) | `conv2d` | `input` (3-d), `conv2d`, `maxpool2d` | 3 | ✅ | ❌ |
| Max-pooling (2-d) | `maxpool2d` | `input` (3-d), `conv2d`, `maxpool2d` | 3 | ✅ | ❌ |
| Flatten | `flatten` | `input` (3-d), `conv2d`, `maxpool2d` | 1 | ✅ | ✅ |

## Getting started

Get the code:

```
git clone https://github.com/modern-fortran/neural-fortran
cd neural-fortran
```

Dependencies:

* Fortran 2018-compatible compiler
* OpenCoarrays (optional, for parallel execution, GFortran only)
* BLAS, MKL (optional)

Compilers tested include:

* gfortran-9.4.0
* ifort-2021.4
* ifx-2021.4

### Building with fpm

#### Building in serial mode

```
fpm build
```

By default, without specifying the build profile, fpm will build neural-fortran
using the debug compiler flags, and without optimization.
To build optimized code, build with the release profile:

```
fpm build --profile release
```

If you're using GFortran, make sure to also pass an additional flag:

```
fpm build --profile release --flag "-fno-frontend-optimize"
```

The `-fno-frontend-optimize` disables some optimizations that may be harmful
when building neural-fortran.

#### Building in parallel mode

If you use GFortran and want to run neural-fortran in parallel,
you must first install [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays).
Once installed, use the compiler wrappers `caf` and `cafrun` to build and execute
in parallel, respectively:

```
fpm build --compiler caf --profile release --flag "-fno-frontend-optimize"
```

#### Testing with fpm

```
fpm test
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

1. [simple](example/simple.f90): Approximating a simple, constant data relationship
2. [sine](example/sine.f90): Approximating a sine function
3. [mnist](example/mnist.f90): Hand-written digit recognition using the MNIST dataset

The MNIST example uses [curl](https://curl.se/) to download the dataset,
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

## Contributors

Thanks to all open-source contributors to neural-fortran:

* [@awvwgk](https://github.com/awvwgk)
* [@ivan-pi](https://github.com/ivan-pi)
* [@jvdp1](https://github.com/jvdp1)
* [@milancurcic](https://github.com/milancurcic)
* [@pirpyn](https://github.com/pirpyn)
* [@rouson](https://github.com/rouson)
* [@scivision](https://github.com/scivision)

## Related projects

* [Fortran Keras Bridge (FKB)](https://github.com/scientific-computing/FKB)
* [rte-rrtmgp](https://github.com/peterukk/rte-rrtmgp)
