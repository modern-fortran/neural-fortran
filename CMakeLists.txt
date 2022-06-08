# CMake version, project name, language
cmake_minimum_required(VERSION 3.20)

# If build type not specified, default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build Release")
endif()

project(neural-fortran
LANGUAGES C Fortran
)

enable_testing()

option(SERIAL "Serial execution")

# Set output paths for modules, archives, and executables
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(SERIAL)
  message(STATUS "Configuring build for serial execution")
else()
  message(STATUS "Configuring build for parallel execution")
endif()

include(FetchContent)

set(h5fortran_BUILD_TESTING false)

FetchContent_Declare(
  h5fortran
  GIT_REPOSITORY https://github.com/geospace-code/h5fortran
  GIT_TAG v4.6.3
  GIT_SHALLOW true
)

FetchContent_Declare(
  jsonfortran
  GIT_REPOSITORY https://github.com/jacobwilliams/json-fortran
  GIT_TAG 8.3.0
  GIT_SHALLOW true
)

FetchContent_MakeAvailable(h5fortran jsonfortran)

file(MAKE_DIRECTORY ${h5fortran_BINARY_DIR}/include)
include_directories(${h5fortran_BINARY_DIR}/include)

file(MAKE_DIRECTORY ${jsonfortran_BINARY_DIR}/include)
include_directories(${jsonfortran_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH ${h5fortran_SOURCE_DIR}/cmake/Modules)
find_package(HDF5 COMPONENTS Fortran REQUIRED)

# compiler flags for gfortran
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")

  if(SERIAL)
    message(STATUS "Configuring to build with -fcoarray=single")
    add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-fcoarray=single>")
  else()
    add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-fcoarray=lib>")
  endif()

  if(BLAS)
    add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-fexternal-blas;${BLAS}>")
    list(APPEND LIBS "blas")
    message(STATUS "Configuring build to use BLAS from ${BLAS}")
  endif()

  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:-fcheck=bounds;-fbacktrace>")
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Release>>:-Ofast;-fno-frontend-optimize;-fno-backtrace>")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "^Intel")
  # compiler flags for ifort

  if(SERIAL)
    message(STATUS "Configuring to build with -coarray=single")
    if(WIN32)
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:/Qcoarray:single>")
    else()
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-coarray=single>")
    endif()
  else()
    if(WIN32)
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:/Qcoarray:shared>")
    else()
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-coarray=shared>")
    endif()
  endif()

  string(APPEND CMAKE_Fortran_FLAGS " -assume byterecl")
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:-check;-traceback>")
  # add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Release>>:-O3>")

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
  # compiler flags for Cray ftn
  string(APPEND CMAKE_Fortran_FLAGS " -h noomp")
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:-O0;-g>")
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Release>>:-O3>")
endif()

# library to archive (libneural.a)
add_library(neural
  src/nf.f90
  src/nf/nf_activation.f90
  src/nf/nf_base_layer.f90
  src/nf/nf_base_layer_submodule.f90
  src/nf/nf_conv2d_layer.f90
  src/nf/nf_conv2d_layer_submodule.f90
  src/nf/nf_datasets.f90
  src/nf/nf_datasets_submodule.f90
  src/nf/nf_datasets_mnist.f90
  src/nf/nf_datasets_mnist_submodule.f90
  src/nf/nf_dense_layer.f90
  src/nf/nf_dense_layer_submodule.f90
  src/nf/nf_flatten_layer.f90
  src/nf/nf_flatten_layer_submodule.f90
  src/nf/nf_input1d_layer.f90
  src/nf/nf_input1d_layer_submodule.f90
  src/nf/nf_input3d_layer.f90
  src/nf/nf_input3d_layer_submodule.f90
  src/nf/nf_keras.f90
  src/nf/nf_keras_submodule.f90
  src/nf/nf_layer_constructors.f90
  src/nf/nf_layer_constructors_submodule.f90
  src/nf/nf_layer.f90
  src/nf/nf_layer_submodule.f90
  src/nf/nf_loss.f90
  src/nf/nf_loss_submodule.f90
  src/nf/nf_maxpool2d_layer.f90
  src/nf/nf_maxpool2d_layer_submodule.f90
  src/nf/nf_network.f90
  src/nf/nf_network_submodule.f90
  src/nf/nf_optimizers.f90
  src/nf/nf_parallel.f90
  src/nf/nf_parallel_submodule.f90
  src/nf/nf_random.f90
  src/nf/nf_random_submodule.f90
  src/nf/io/nf_io_binary.f90
  src/nf/io/nf_io_binary_submodule.f90
  src/nf/io/nf_io_hdf5.f90
  src/nf/io/nf_io_hdf5_submodule.f90
)
target_link_libraries(neural PRIVATE h5fortran::h5fortran HDF5::HDF5)
target_include_directories(neural PRIVATE ${jsonfortran_BINARY_DIR}/include)

# Remove leading or trailing whitespace
string(REGEX REPLACE "^ | $" "" LIBS "${LIBS}")

# tests

foreach(execid input1d_layer input3d_layer dense_layer conv2d_layer maxpool2d_layer flatten_layer dense_network dense_network_from_keras conv2d_network io_hdf5 keras_read_model)
  add_executable(test_${execid} test/test_${execid}.f90)
  target_link_libraries(test_${execid} PRIVATE neural h5fortran::h5fortran jsonfortran ${LIBS})
  add_test(test_${execid} bin/test_${execid})
endforeach()

foreach(execid cnn mnist mnist_from_keras simple sine)
  add_executable(${execid} example/${execid}.f90)
  target_link_libraries(${execid} PRIVATE neural h5fortran::h5fortran jsonfortran ${LIBS})
endforeach()
