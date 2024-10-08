# compiler flags for gfortran
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")

  if(PARALLEL)
    message(STATUS "Configuring to build with -fcoarray=shared")
    add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-fcoarray=shared>")
    add_compile_definitions(PARALLEL)
  else()
    add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-fcoarray=single>")
  endif()

  if(BLAS)
    add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-fexternal-blas;${BLAS}>")
    list(APPEND LIBS "blas")
    message(STATUS "Configuring build to use BLAS from ${BLAS}")
  endif()

  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:-cpp;-fcheck=bounds;-fbacktrace>")
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Release>>:-cpp;-Ofast;-fno-frontend-optimize;-fno-backtrace>")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "^Intel")
  # compiler flags for ifort

  if(PARALLEL)
    message(STATUS "Configuring to build with -coarray=shared")
    if(WIN32)
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:/Qcoarray:shared>")
      add_link_options("$<$<COMPILE_LANGUAGE:Fortran>:/Qcoarray:shared>")
    else()
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-coarray=shared>")
      add_link_options("$<$<COMPILE_LANGUAGE:Fortran>:-coarray=shared>")
    endif()
    add_compile_definitions(PARALLEL)
  else()
    if(WIN32)
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:/Qcoarray:shared>")
      add_link_options("$<$<COMPILE_LANGUAGE:Fortran>:/Qcoarray:shared>")
    else()
      add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:-coarray=shared>")
      add_link_options("$<$<COMPILE_LANGUAGE:Fortran>:-coarray=shared>")
    endif()
  endif()

  if(WIN32)
    string(APPEND CMAKE_Fortran_FLAGS " /assume:byterecl /fpp")
  else()
    string(APPEND CMAKE_Fortran_FLAGS " -assume byterecl -fpp")
  endif()
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:-fpp;-check;-traceback>")
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Release>>:-fpp;-O3>")

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
  # compiler flags for Cray ftn
  string(APPEND CMAKE_Fortran_FLAGS " -h noomp")
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:-e Z;-O0;-g>")
  add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Release>>:-e Z;-O3>")
endif()
