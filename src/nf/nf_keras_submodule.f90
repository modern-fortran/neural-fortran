submodule(nf_keras) nf_keras_submodule

  use json_module, only: json_core, json_value
  use nf_io_hdf5, only: get_h5_attribute_string

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
    integer :: n, num_layers, num_elements
    logical :: found

    model_config_string = get_h5_attribute_string(filename, '.', 'model_config')

    call json % parse(model_config_json, model_config_string)
    call json % get(model_config_json, 'config.layers', layers_json)

    num_layers = json % count(layers_json)

    allocate(res(num_layers))

    ! Iterate over layers
    layers: do n = 1, num_layers

      ! Get pointer to the layer
      call json % get_child(layers_json, n, layer_json)

      ! Get type of layer as a string
      call json % get(layer_json, 'class_name', res(n) % type)

      ! Get pointer to the layer config
      call json % get(layer_json, 'config', layer_config_json)

      ! Get size of layer and activation if applicable;
      ! Instantiate neural-fortran layers at this time.
      if (res(n) % type == 'InputLayer') then
        
        call json % get(layer_config_json, 'batch_input_shape', tmp_array)
        res(n) % num_elements = [tmp_array(2)]
      
      else if (res(n) % type == 'Dense') then
        
        call json % get(layer_config_json, 'units',  num_elements, found)
        res(n) % num_elements = [num_elements]

        call json % get(layer_config_json, 'activation', res(n) % activation)
      
      else
        
        error stop 'This layer is not supported'
      
      end if

    end do layers

  end function get_keras_h5_layers

end submodule nf_keras_submodule
