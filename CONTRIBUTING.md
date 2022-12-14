# Contributing guide

This document describes the organization of the neural-fortran codebase to help
guide the code contributors.

## Overall code organization

The source code organization follows the usual `fpm` convention:
the library code is in [src/](src/), test programs are in [test/](test/),
and example programs are in [example/](example/).

The top-level module that suggests the public, user-facing API is in
[src/nf.f90](src/nf.f90).
All other library source files are in [src/nf/](src/nf/).

Most of the library code defines interfaces in modules and implementations in
submodules.
If you want to know only about interfaces, in other words how to call things,
you can read just the module source files and not worry about the implementation.
Then, if you want to know more about the implementation, you can find it in the
appropriate source file that defines the submodule.
Each library source file contains either one module or one submodule.
The source files that define the submodule end with `_submodule.f90`.

## Components

Neural-fortran defines several components, described in a roughly top-down order:

* Networks
* Layers
  - Layer constructor functions
  - Concrete layer implementations
* Optimizers
* Activation functions

### Networks

A network is the main component that the user works with,
and the highest-level container in neural-fortran.
A network is a collection of layers.

The network container is defined by the `network` derived type
in the `nf_network` module, in the [nf_network.f90](src/nf/nf_network.f90)
source file.

In a nutshell, the `network` type defines an allocatable array of `type(layer)`
instances, and several type-bound methods for training and inference.

### Layers

### Optimizers

### Activation functions