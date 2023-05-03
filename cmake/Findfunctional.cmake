# This is a CMake file to detect the installation of the functional-fortran library
# An install of this library provides:
#
# 1. A library named "libfunctional.a"
# 2. A variety of Fortran .mod files

include(FindPackageHandleStandardArgs)

find_library(functional_LIBRARY
  NAMES functional)

find_path(functional_INCLUDE_DIR
  NAMES functional.mod)

find_package_handle_standard_args(functional DEFAULT_MSG
  functional_LIBRARY functional_INCLUDE_DIR)

if (functional_FOUND)
  mark_as_advanced(functional_INCLUDE_DIR)
  mark_as_advanced(functional_LIBRARY)
endif()

if (functional_FOUND AND NOT functional::functional)
  add_library(functional::functional IMPORTED STATIC)
  set_property(TARGET functional::functional PROPERTY IMPORTED_LOCATION ${functional_LIBRARY})
  target_include_directories(functional::functional INTERFACE ${functional_INCLUDE_DIR})
endif()
