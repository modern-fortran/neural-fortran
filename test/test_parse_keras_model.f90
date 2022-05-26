program test_parse_keras_model

  use nf_io_hdf5, only: get_h5_attribute_string
  use json_module

  implicit none

  character(:), allocatable :: model_config_string
  character(*), parameter :: test_data_path = 'test/data/mnist_dense.h5'
  type(json_core) :: json
  type(json_value), pointer :: model_config, layers, next_layer, layer
  character(:), allocatable :: class_name, layer_type
  logical :: found
  integer :: n, num_layers

  model_config_string = &
    get_h5_attribute_string(test_data_path, '.', 'model_config')

  call json % parse(model_config, model_config_string)
  call json % get(model_config, 'config.layers', layers)

  num_layers = json % count(layers)
  print *, 'This model has', num_layers, 'layers.'

  do n = 1, num_layers
    call json % get_child(layers, n, layer)
    print *, 'Layer', n
    !call json % print(layer)
    call json % get(layer, 'class_name', layer_type)
    print *, layer_type
  end do

end program test_parse_keras_model
