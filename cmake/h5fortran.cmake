find_package(HDF5 COMPONENTS Fortran REQUIRED
  OPTIONAL_COMPONENTS C HL)
if (HDF5_HL_FOUND)
  message(STATUS "HDF5 HL is available")
  target_link_libraries(HDF5::HDF5 INTERFACE hdf5::hdf5_hl hdf5::hdf5_hl_fortran)
endif()
if (HDF5_IS_PARALLEL)
  message(STATUS "HDF5 is parallel")
  find_package(MPI REQUIRED COMPONENTS Fortran)
  target_link_libraries(HDF5::HDF5 INTERFACE MPI::MPI_Fortran)
endif()

find_package(h5fortran 4.6.3 QUIET)

if (NOT h5fortran_FOUND)
  message(STATUS "h5fortran not found, fetching from GitHub")

  set(h5fortran_BUILD_TESTING false)

  FetchContent_Declare(h5fortran
    GIT_REPOSITORY https://github.com/geospace-code/h5fortran
    GIT_TAG v4.6.3
    GIT_SHALLOW true
  )

  FetchContent_MakeAvailable(h5fortran)

  file(MAKE_DIRECTORY ${h5fortran_BINARY_DIR}/include)


  list(APPEND CMAKE_MODULE_PATH ${h5fortran_SOURCE_DIR}/cmake/Modules)
else()
  message(STATUS "h5fortran found: ${h5fortran_DIR}")
endif()
