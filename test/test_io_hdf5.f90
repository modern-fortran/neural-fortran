program test_io_hdf5

  use nf_io_hdf5, only: get_h5_attribute_string
  implicit none

  print *, get_h5_attribute_string('test/data/mnist_dense.h5', '.', 'model_config')
  print *, get_h5_attribute_string('test/data/mnist_dense.h5', 'model_weights', 'layer_names')

end program test_io_hdf5
