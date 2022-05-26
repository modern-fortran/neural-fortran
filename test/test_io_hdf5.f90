program test_io_hdf5

  use nf_io_hdf5, only: get_h5_attribute_string

  implicit none

  character(:), allocatable :: attr
  character(*), parameter :: test_data_path = 'test/data/mnist_dense.h5'

  attr = get_h5_attribute_string(test_data_path, '.', 'backend')

end program test_io_hdf5
