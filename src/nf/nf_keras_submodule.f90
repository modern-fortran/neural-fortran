submodule(nf_keras) nf_keras_submodule

  use functional, only: reverse
  use json_module, only: json_core, json_value
  use nf_io_hdf5, only: hdf5_attribute_string

  implicit none

contains

  module function get_keras_h5_layers(filename) result(res)
    character(*), intent(in) :: filename
    type(keras_layer), allocatable :: res(:)

    character(:), allocatable :: model_config_string

    type(json_core) :: json
    type(json_value), pointer :: &
      model_config_json, layers_json, layer_json, layer_config_json

    real, allocatable :: tmp_array(:)
    integer :: n, num_layers, units
    logical :: found

    model_config_string = hdf5_attribute_string(filename, '.', 'model_config')

    call json % parse(model_config_json, model_config_string)
    call json % get(model_config_json, 'config.layers', layers_json)

    num_layers = json % count(layers_json)

    allocate(res(num_layers))

    ! Iterate over layers
    layers: do n = 1, num_layers

      ! Get pointer to the layer
      call json % get_child(layers_json, n, layer_json)

      ! Get type of layer as a string
      call json % get(layer_json, 'class_name', res(n) % class)

      ! Get pointer to the layer config
      call json % get(layer_json, 'config', layer_config_json)

      ! Get layer name
      call json % get(layer_config_json, 'name', res(n) % name)

      ! Get size of layer and activation if applicable;
      ! Instantiate neural-fortran layers at this time.
      select case(res(n) % class)

        case('InputLayer')
          call json % get(layer_config_json, 'batch_input_shape', tmp_array)
          res(n) % units = reverse(tmp_array(2:)) ! skip the 1st (batch) dim
      
        case('Dense')
          call json % get(layer_config_json, 'units', units, found)
          res(n) % units = [units]
          call json % get(layer_config_json, 'activation', res(n) % activation)

        case('Flatten')
          ! Nothing to read here; merely a placeholder.
          continue

        case('Conv2D')
          call json % get(layer_config_json, &
            'filters', res(n) % filters, found)
          call json % get(layer_config_json, &
            'kernel_size', res(n) % kernel_size, found)
          call json % get(layer_config_json, &
            'activation', res(n) % activation)
          ! Reverse to account for C -> Fortran order
          res(n) % kernel_size = reverse(res(n) % kernel_size)

        case('MaxPooling2D')
          call json % get(layer_config_json, &
            'pool_size',  res(n) % pool_size, found)
          call json % get(layer_config_json, &
            'strides',  res(n) % strides, found)
          ! Reverse to account for C -> Fortran order
          res(n) % pool_size = reverse(res(n) % pool_size)
          res(n) % strides = reverse(res(n) % strides)

        case('Reshape')
          ! Only read target shape
          call json % get(layer_config_json, &
            'target_shape',  res(n) % target_shape, found)
          ! Reverse to account for C -> Fortran order
          res(n) % target_shape = reverse(res(n) % target_shape)

        case default
          error stop 'This Keras layer is not supported'

      end select

    end do layers

    ! free the memory:
    call json % destroy(model_config_json)

  end function get_keras_h5_layers

end submodule nf_keras_submodule
