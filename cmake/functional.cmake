FetchContent_Declare(functional
  GIT_REPOSITORY https://github.com/wavebitscientific/functional-fortran
  GIT_TAG 0.6.1
  GIT_SHALLOW true
)

FetchContent_Populate(functional)

add_library(functional ${functional_SOURCE_DIR}/src/functional.f90)
target_include_directories(functional PUBLIC
$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
$<INSTALL_INTERFACE:include>
)

add_library(functional::functional INTERFACE IMPORTED GLOBAL)
target_link_libraries(functional::functional INTERFACE functional)

install(TARGETS functional)
