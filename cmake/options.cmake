option(SERIAL "Serial execution")
option(${PROJECT_NAME}_BUILD_TESTING "build ${PROJECT_NAME} tests" true)
option(${PROJECT_NAME}_BUILD_EXAMPLES "build ${PROJECT_NAME} examples" true)

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

# --- Generally useful CMake project options

# Rpath options necessary for shared library install to work correctly in user projects
set(CMAKE_INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)

# Necessary for shared library with Visual Studio / Windows oneAPI
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS true)

# --- auto-ignore build directory
if(NOT EXISTS ${PROJECT_BINARY_DIR}/.gitignore)
  file(WRITE ${PROJECT_BINARY_DIR}/.gitignore "*")
endif()
