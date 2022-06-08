set(h5fortran_BUILD_TESTING false)

FetchContent_Declare(h5fortran
  GIT_REPOSITORY https://github.com/geospace-code/h5fortran
  GIT_TAG v4.6.3
  GIT_SHALLOW true
)

FetchContent_MakeAvailable(h5fortran)

file(MAKE_DIRECTORY ${h5fortran_BINARY_DIR}/include)


list(APPEND CMAKE_MODULE_PATH ${h5fortran_SOURCE_DIR}/cmake/Modules)
find_package(HDF5 COMPONENTS Fortran REQUIRED)
