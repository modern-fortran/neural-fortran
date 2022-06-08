# CMake version, project name, language
cmake_minimum_required(VERSION 3.20)
project(neural-fortran Fortran)

# Set output paths for modules, archives, and executables
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# If build type not specified, default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "release")
endif()

if(SERIAL)
  message(STATUS "Configuring build for serial execution")
else()
  message(STATUS "Configuring build for parallel execution")
endif()

find_package(HDF5 COMPONENTS Fortran HL)
include_directories(${HDF5_INCLUDE_DIRS})

include(FetchContent)

FetchContent_Declare(
  h5fortran
  GIT_REPOSITORY https://github.com/geospace-code/h5fortran
  GIT_TAG v4.6.3
)

FetchContent_Declare(
  jsonfortran
  GIT_REPOSITORY https://github.com/jacobwilliams/json-fortran
  GIT_TAG 8.3.0
)

FetchContent_MakeAvailable(h5fortran jsonfortran)

file(MAKE_DIRECTORY ${h5fortran_BINARY_DIR}/include)
include_directories(${h5fortran_BINARY_DIR}/include)

file(MAKE_DIRECTORY ${jsonfortran_BINARY_DIR}/include)
include_directories(${jsonfortran_BINARY_DIR})

# compiler flags for gfortran
if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)

  if(SERIAL)
    message(STATUS "Configuring to build with -fcoarray=single")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fcoarray=single")
  endif()

  if(BLAS)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fexternal-blas ${BLAS}")
    set(LIBS "${LIBS} blas")
    message(STATUS "Configuring build to use BLAS from ${BLAS}")
  endif()
    
  set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -fcheck=bounds -fbacktrace")
  set(CMAKE_Fortran_FLAGS_RELEASE "-Ofast -fno-frontend-optimize")
endif()

# compiler flags for ifort
if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)

  if(SERIAL)
    message(STATUS "Configuring to build with -coarray=single")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -coarray=single")
  endif()

  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -assume byterecl")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -C -traceback")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3")

  if(NOT SERIAL)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -coarray=shared")
  endif()

endif()

# compiler flags for Cray ftn
if(CMAKE_Fortran_COMPILER_ID MATCHES Cray)
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -h noomp")
  set(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_Fortran_FLAGS_RELEASE "-O3")
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

# Remove leading or trailing whitespace
string(REGEX REPLACE "^ | $" "" LIBS "${LIBS}")

# tests
enable_testing()
foreach(execid input1d_layer input3d_layer dense_layer conv2d_layer maxpool2d_layer flatten_layer dense_network dense_network_from_keras conv2d_network io_hdf5 keras_read_model)
  add_executable(test_${execid} test/test_${execid}.f90)
  target_link_libraries(test_${execid} PRIVATE neural h5fortran::h5fortran jsonfortran ${LIBS})
  add_test(test_${execid} bin/test_${execid})
endforeach()
  
foreach(execid cnn mnist mnist_from_keras simple sine)
  add_executable(${execid} example/${execid}.f90)
  target_link_libraries(${execid} PRIVATE neural h5fortran::h5fortran jsonfortran ${LIBS})
endforeach()
