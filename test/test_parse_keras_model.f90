program test_parse_keras_model

  use iso_fortran_env, only: stderr => error_unit
  use nf_datasets, only: download_and_unpack, keras_model_dense_mnist_url
  use nf_io_hdf5, only: get_h5_attribute_string
  use json_module

  implicit none

  character(:), allocatable :: model_config_string
  character(*), parameter :: test_data_path = 'keras_dense_mnist.h5'
  type(json_core) :: json
  type(json_value), pointer :: model_config, layers, next_layer, layer
  character(:), allocatable :: class_name, layer_type
  logical :: found
  integer :: n, num_layers
  logical :: file_exists
  logical :: ok = .true.

  inquire(file=test_data_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_model_dense_mnist_url)

  model_config_string = &
    get_h5_attribute_string(test_data_path, '.', 'model_config')

  call json % parse(model_config, model_config_string)
  call json % get(model_config, 'config.layers', layers)

  num_layers = json % count(layers)

  do n = 1, num_layers
    call json % get_child(layers, n, layer)
    !print *, 'Layer', n
    !call json % print(layer)
    call json % get(layer, 'class_name', layer_type)
    !print *, layer_type
  end do

  if (.not. num_layers == 3) then
    ok = .false.
    write(stderr, '(a)') 'Keras dense MNIST model has 3 layers.. failed'
  end if

  if (ok) then
    print '(a)', 'test_parse_keras_model: All tests passed.'
  else
    write(stderr, '(a)') 'test_parse_keras_model: One or more tests failed.'
    stop 1
  end if

end program test_parse_keras_model
