program test_io_h5

  use h5fortran, only : hdf5_file
  implicit none

  type(hdf5_file) :: h5f

  call h5f % open('test_file.h5', action='w')
  call h5f % write('/x', 123)
  call h5f % close()

end program test_io_h5
