program test_parse_keras_model

  use iso_fortran_env, only: stderr => error_unit
  use nf_datasets, only: download_and_unpack, keras_model_dense_mnist_url
  use nf_io_hdf5, only: get_h5_attribute_string
  use nf, only: layer, network, dense, input
  use json_module

  implicit none

  character(:), allocatable :: model_config_string
  character(*), parameter :: test_data_path = 'keras_dense_mnist.h5'
  type(json_core) :: json
  type(json_value), pointer :: &
    model_config, layer_list, this_layer, layer_config
  character(:), allocatable :: class_name, layer_type, activation
  real, allocatable :: tmp_array(:)

  type(layer), allocatable :: layers(:)
  type(network) :: net

  logical :: found
  integer :: n, num_layers, num_elements
  logical :: file_exists
  logical :: ok = .true.

  inquire(file=test_data_path, exist=file_exists)
  if (.not. file_exists) call download_and_unpack(keras_model_dense_mnist_url)

  model_config_string = &
    get_h5_attribute_string(test_data_path, '.', 'model_config')

  call json % parse(model_config, model_config_string)
  call json % get(model_config, 'config.layers', layer_list)

  num_layers = json % count(layer_list)

  layers = [layer ::]

  ! Iterate over layers
  do n = 1, num_layers

    ! Get pointer to the layer
    call json % get_child(layer_list, n, this_layer)

    ! Get type of layer as a string
    call json % get(this_layer, 'class_name', layer_type)

    ! Get pointer to the layer config
    call json % get(this_layer, 'config', layer_config)

    ! Get size of layer and activation if applicable;
    ! Instantiate neural-fortran layers at this time.
    if (layer_type == 'InputLayer') then
      call json % get(layer_config, 'batch_input_shape', tmp_array, found)
      num_elements = tmp_array(2)
      layers = [layers, input(num_elements)]
    else if (layer_type == 'Dense') then
      call json % get(layer_config, 'units', num_elements, found)
      call json % get(layer_config, 'activation', activation, found)
      layers = [layers, dense(num_elements, activation)]
    else
      error stop 'This layer is not supported'
    end if

    print *, n, layer_type, num_elements, activation
  end do

  net = network(layers)
  call net % print_info()

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
