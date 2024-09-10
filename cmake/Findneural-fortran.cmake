# Find the native neural-fortran includes and library
#
# neural-fortran_INCLUDE_DIRS - where to find neural-fortran.h, etc.
# neural-fortran_LIBRARIES - List of libraries when using neural-fortran.
# neural-fortran_FOUND - True if neural-fortran found.
#
# To use neural-fortran_ROOT_DIR to specify the prefix directory of neural-fortran


find_path(neural-fortran_INCLUDE_DIRS
  NAMES nf.mod
  HINTS ${neural-fortran_ROOT_DIR}/include ENV neural-fortran_INCLUDE_DIR)

find_library(neural-fortran_LIBRARIES
  NAMES neural-fortran
  HINTS ${neural-fortran_ROOT_DIR}/lib ENV neural-fortran_LIB_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(neural-fortran DEFAULT_MSG neural-fortran_LIBRARIES neural-fortran_INCLUDE_DIRS)

mark_as_advanced(
  neural-fortran_LIBRARIES
  neural-fortran_INCLUDE_DIRS)

if(neural-fortran_FOUND AND NOT (TARGET neural-fortran::neural-fortran))
  add_library (neural-fortran::neural-fortran STATIC IMPORTED)
  set_target_properties(neural-fortran::neural-fortran
    PROPERTIES
      IMPORTED_LOCATION ${neural-fortran_LIBRARIES}
      INTERFACE_INCLUDE_DIRECTORIES ${neural-fortran_INCLUDE_DIRS})
endif()
