# Per the jsonfortran README, if you build with their CMakeLists.txt, you will
# make a jsonfortran-${CMAKE_Fortran_COMPILER_ID} package, so you have to find_package
# on that.
find_package(jsonfortran-${CMAKE_Fortran_COMPILER_ID} 8.3.0 QUIET)

if (NOT jsonfortran-${CMAKE_Fortran_COMPILER_ID}_FOUND)
  message(STATUS "jsonfortran not found, fetching from github")

  # use our own CMake script to build jsonfortran instead of jsonfortran/CMakelists.txt

  FetchContent_Declare(jsonfortran
    GIT_REPOSITORY https://github.com/jacobwilliams/json-fortran
    GIT_TAG 8.3.0
    GIT_SHALLOW true
  )

  FetchContent_Populate(jsonfortran)

  SET(JSON_REAL_KIND "REAL64")
  SET(JSON_INT_KIND "INT32")

  set(_src ${jsonfortran_SOURCE_DIR}/src)

  set (JF_LIB_SRCS
  ${_src}/json_kinds.F90
  ${_src}/json_parameters.F90
  ${_src}/json_string_utilities.F90
  ${_src}/json_value_module.F90
  ${_src}/json_file_module.F90
  ${_src}/json_module.F90
  )

  add_library(jsonfortran ${JF_LIB_SRCS})
  target_compile_definitions(jsonfortran PRIVATE ${JSON_REAL_KIND} ${JSON_INT_KIND})
  target_include_directories(jsonfortran PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )

  add_library(jsonfortran::jsonfortran INTERFACE IMPORTED GLOBAL)
  target_link_libraries(jsonfortran::jsonfortran INTERFACE jsonfortran)

  install(TARGETS jsonfortran)
else ()
  message(STATUS "jsonfortran found: ${jsonfortran-${CMAKE_Fortran_COMPILER_ID}_DIR}")
  # We need to make a jsonfortran::jsonfortran target for the jsonfortran we found because
  # neural-fortran is expecting it. It looks like it is a lower-case version of the compiler
  # name, so we'll just do that.
  string(TOLOWER ${CMAKE_Fortran_COMPILER_ID} lower_compiler_id)
  add_library(jsonfortran::jsonfortran ALIAS jsonfortran-${lower_compiler_id}::jsonfortran)
endif ()
