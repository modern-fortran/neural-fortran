option(SERIAL "Serial execution")

# Set output paths for modules, archives, and executables
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(SERIAL)
  message(STATUS "Configuring build for serial execution")
else()
  message(STATUS "Configuring build for parallel execution")
endif()
