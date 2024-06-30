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

include(FetchContent)

include(cmake/options.cmake)
include(cmake/compilers.cmake)

include(cmake/functional.cmake)
include(cmake/h5fortran.cmake)
include(cmake/json.cmake)

# library to archive (libneural-fortran.a)
add_library(neural-fortran
  src/nf.f90
  src/nf/nf_activation.f90
  src/nf/nf_base_layer.f90
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
  src/nf/nf_metrics.f90
  src/nf/nf_network.f90
  src/nf/nf_network_submodule.f90
  src/nf/nf_optimizers.f90
  src/nf/nf_parallel.f90
  src/nf/nf_parallel_submodule.f90
  src/nf/nf_random.f90
  src/nf/nf_reshape_layer.f90
  src/nf/nf_reshape_layer_submodule.f90
  src/nf/nf_rnn_layer.f90
  src/nf/nf_rnn_layer_submodule.f90
  src/nf/io/nf_io_binary.f90
  src/nf/io/nf_io_binary_submodule.f90
  src/nf/io/nf_io_hdf5.f90
  src/nf/io/nf_io_hdf5_submodule.f90
)

target_link_libraries(neural-fortran PRIVATE
  functional::functional
  h5fortran::h5fortran
  HDF5::HDF5
  jsonfortran::jsonfortran
)

install(TARGETS neural-fortran)

# Remove leading or trailing whitespace
string(REGEX REPLACE "^ | $" "" LIBS "${LIBS}")

if(${PROJECT_NAME}_BUILD_TESTING)
  add_subdirectory(test)
endif()

if(${PROJECT_NAME}_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()
