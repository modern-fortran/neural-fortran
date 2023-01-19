submodule(nf_network) nf_network_submodule

  use nf_conv2d_layer, only: conv2d_layer
  use nf_dense_layer, only: dense_layer
  use nf_flatten_layer, only: flatten_layer
  use nf_input1d_layer, only: input1d_layer
  use nf_input3d_layer, only: input3d_layer
  use nf_maxpool2d_layer, only: maxpool2d_layer
  use nf_reshape_layer, only: reshape3d_layer
  use nf_io_hdf5, only: get_hdf5_dataset
  use nf_keras, only: get_keras_h5_layers, keras_layer
  use nf_layer, only: layer
  use nf_layer_constructors, only: conv2d, dense, flatten, input, maxpool2d, reshape
  use nf_loss, only: quadratic_derivative
  use nf_optimizers, only: optimizer_base_type, sgd
  use nf_parallel, only: tile_indices

  implicit none

contains

  module function network_from_layers(layers) result(res)
    type(layer), intent(in) :: layers(:)
    type(network) :: res
    integer :: n

    ! Error handling

    ! There must be at least two layers
    if (size(layers) < 2) &
      error stop 'Error: A network must have at least 2 layers.'

    ! The first layer must be an input layer
    if (.not. layers(1) % name == 'input') &
      error stop 'Error: First layer in the network must be an input layer.'

    !TODO Ensure that the layers are in allowed sequence:
    !TODO   input1d -> dense
    !TODO   dense -> dense
    !TODO   input3d -> conv2d, maxpool2d, flatten
    !TODO   conv2d -> conv2d, maxpool2d, flatten
    !TODO   maxpool2d -> conv2d, maxpool2d, flatten
    !TODO   flatten -> dense
    !TODO   reshape -> conv2d, maxpool2d

    res % layers = layers

    ! Loop over each layer in order and call the init methods.
    ! This will allocate the data internal to each layer (e.g. weights, biases)
    ! according to the size of the previous layer.
    do n = 2, size(layers)
      call res % layers(n) % init(res % layers(n - 1))
    end do

  end function network_from_layers


  module function network_from_keras(filename) result(res)
    character(*), intent(in) :: filename
    type(network) :: res
    type(keras_layer), allocatable :: keras_layers(:)
    type(layer), allocatable :: layers(:)
    character(:), allocatable :: layer_name
    character(:), allocatable :: object_name
    integer :: n

    keras_layers = get_keras_h5_layers(filename)

    allocate(layers(size(keras_layers)))

    do n = 1, size(layers)

      select case(keras_layers(n) % class)

        case('Conv2D')

          if (keras_layers(n) % kernel_size(1) &
            /= keras_layers(n) % kernel_size(2)) &
            error stop 'Non-square kernel in conv2d layer not supported.'

          layers(n) = conv2d( &
            keras_layers(n) % filters, &
            !FIXME add support for non-square kernel
            keras_layers(n) % kernel_size(1), &
            keras_layers(n) % activation &
          )

        case('Dense')
          layers(n) = dense( &
            keras_layers(n) % units(1), &
            keras_layers(n) % activation &
          )

        case('Flatten')
          layers(n) = flatten()

        case('InputLayer')
          if (size(keras_layers(n) % units) == 1) then
            ! input1d
            layers(n) = input(keras_layers(n) % units(1))
          else
            ! input3d
            layers(n) = input(keras_layers(n) % units)
          end if

        case('MaxPooling2D')

          if (keras_layers(n) % pool_size(1) &
            /= keras_layers(n) % pool_size(2)) &
            error stop 'Non-square pool in maxpool2d layer not supported.'

          if (keras_layers(n) % strides(1) &
            /= keras_layers(n) % strides(2)) &
            error stop 'Unequal strides in maxpool2d layer are not supported.'

          layers(n) = maxpool2d( &
            !FIXME add support for non-square pool and stride
            keras_layers(n) % pool_size(1), &
            keras_layers(n) % strides(1) &
          )

        case('Reshape')
          layers(n) = reshape(keras_layers(n) % target_shape)

        case default
          error stop 'This Keras layer is not supported'

      end select

    end do

    res = network(layers)

    ! Loop over layers and read weights and biases from the Keras h5 file
    ! for each; currently only dense layers are implemented.
    do n = 2, size(res % layers)

      layer_name = keras_layers(n) % name

      select type(this_layer => res % layers(n) % p)

        type is(conv2d_layer)
          ! Read biases from file
          object_name = '/model_weights/' // layer_name // '/' &
            // layer_name // '/bias:0'
          call get_hdf5_dataset(filename, object_name, this_layer % biases)

          ! Read weights from file
          object_name = '/model_weights/' // layer_name // '/' &
            // layer_name // '/kernel:0'
          call get_hdf5_dataset(filename, object_name, this_layer % kernel)

        type is(dense_layer)

          ! Read biases from file
          object_name = '/model_weights/' // layer_name // '/' &
            // layer_name // '/bias:0'
          call get_hdf5_dataset(filename, object_name, this_layer % biases)

          ! Read weights from file
          object_name = '/model_weights/' // layer_name // '/' &
            // layer_name // '/kernel:0'
          call get_hdf5_dataset(filename, object_name, this_layer % weights)

        type is(flatten_layer)
          ! Nothing to do
          continue

        type is(maxpool2d_layer)
          ! Nothing to do
          continue

        type is(reshape3d_layer)
          ! Nothing to do
          continue

        class default
          error stop 'Internal error in network_from_keras(); ' &
            // 'mismatch in layer types between the Keras and ' &
            // 'neural-fortran model layers.'

      end select

  end do

  end function network_from_keras


  pure module subroutine backward(self, output)
    class(network), intent(in out) :: self
    real, intent(in) :: output(:)
    real, allocatable :: gradient(:)
    integer :: n, num_layers

    num_layers = size(self % layers)

    ! Iterate backward over layers, from the output layer
    ! to the first non-input layer
    do n = num_layers, 2, -1

      if (n == num_layers) then
        ! Output layer; apply the loss function
        select type(this_layer => self % layers(n) % p)
          type is(dense_layer)
            gradient = quadratic_derivative(output, this_layer % output)
        end select
      else
        ! Hidden layer; take the gradient from the next layer
        select type(next_layer => self % layers(n + 1) % p)
          type is(dense_layer)
            gradient = next_layer % gradient
        end select
      end if

      call self % layers(n) % backward(self % layers(n - 1), gradient)

    end do

  end subroutine backward


  pure module subroutine forward_1d(self, input)
    class(network), intent(in out) :: self
    real, intent(in) :: input(:)
    integer :: n

    ! Set the input array into the input layer
    select type(input_layer => self % layers(1) % p); type is(input1d_layer)
      call input_layer % set(input)
    end select

    do n = 2, size(self % layers)
      call self % layers(n) % forward(self % layers(n - 1))
    end do

  end subroutine forward_1d


  pure module subroutine forward_3d(self, input)
    class(network), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    integer :: n

    ! Set the input array into the input layer
    select type(input_layer => self % layers(1) % p); type is(input3d_layer)
      call input_layer % set(input)
    end select

    do n = 2, size(self % layers)
      call self % layers(n) % forward(self % layers(n - 1))
    end do

  end subroutine forward_3d


  module function predict_1d(self, input) result(res)
    class(network), intent(in out) :: self
    real, intent(in) :: input(:)
    real, allocatable :: res(:)
    integer :: num_layers

    num_layers = size(self % layers)

    call self % forward(input)

    select type(output_layer => self % layers(num_layers) % p)
      type is(dense_layer)
        res = output_layer % output
      type is(flatten_layer)
        res = output_layer % output
      class default
        error stop 'network % output not implemented for this output layer'
    end select

  end function predict_1d


  module function predict_3d(self, input) result(res)
    class(network), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, allocatable :: res(:)
    integer :: num_layers

    num_layers = size(self % layers)

    call self % forward(input)

    select type(output_layer => self % layers(num_layers) % p)
      type is(conv2d_layer)
        !FIXME flatten the result for now; find a better solution
        res = pack(output_layer % output, .true.)
      type is(dense_layer)
        res = output_layer % output
      type is(flatten_layer)
        res = output_layer % output
      class default
        error stop 'network % output not implemented for this output layer'
    end select

  end function predict_3d


  module function predict_batch_1d(self, input) result(res)
    class(network), intent(in out) :: self
    real, intent(in) :: input(:,:)
    real, allocatable :: res(:,:)
    integer :: i, batch_size, num_layers, output_size

    num_layers = size(self % layers)
    batch_size = size(input, dim=rank(input))
    output_size = product(self % layers(num_layers) % layer_shape)

    allocate(res(output_size, batch_size))

    batch: do concurrent(i = 1:size(res, dim=2))

      call self % forward(input(:,i))

      select type(output_layer => self % layers(num_layers) % p)
        type is(dense_layer)
          res(:,i) = output_layer % output
        type is(flatten_layer)
          res(:,i) = output_layer % output
        class default
          error stop 'network % output not implemented for this output layer'
      end select

    end do batch

  end function predict_batch_1d


  module function predict_batch_3d(self, input) result(res)
    class(network), intent(in out) :: self
    real, intent(in) :: input(:,:,:,:)
    real, allocatable :: res(:,:)
    integer :: i, batch_size, num_layers, output_size

    num_layers = size(self % layers)
    batch_size = size(input, dim=rank(input))
    output_size = product(self % layers(num_layers) % layer_shape)

    allocate(res(output_size, batch_size))

    batch: do concurrent(i = 1:batch_size)

      call self % forward(input(:,:,:,i))

      select type(output_layer => self % layers(num_layers) % p)
        type is(conv2d_layer)
          !FIXME flatten the result for now; find a better solution
          res(:,i) = pack(output_layer % output, .true.)
        type is(dense_layer)
          res(:,i) = output_layer % output
        type is(flatten_layer)
          res(:,i) = output_layer % output
        class default
          error stop 'network % output not implemented for this output layer'
      end select

    end do batch

  end function predict_batch_3d


  module subroutine print_info(self)
    class(network), intent(in) :: self
    call self % layers % print_info()
  end subroutine print_info


  pure module function get_num_params(self)
    class(network), intent(in) :: self
    integer :: get_num_params

    get_num_params = sum(self % layers % get_num_params())

  end function get_num_params


  pure module function get_params(self) result(params)
    class(network), intent(in) :: self
    real, allocatable :: params(:)
    integer :: n, nstart, nend

    allocate(params(self % get_num_params()))

    nstart = 1
    do n = 1, size(self % layers)

      if (self % layers(n) % get_num_params() < 1) cycle

      nend = nstart + self % layers(n) % get_num_params() - 1
      params(nstart:nend) = self % layers(n) % get_params()
      nstart = nend + 1
    end do

  end function get_params


  module subroutine set_params(self, params)
    class(network), intent(in out) :: self
    real, intent(in) :: params(:)
    integer :: n, nstart, nend

    ! Check that the number of parameters is correct.
    if (size(params) /= self % get_num_params()) then
      error stop 'network % set_params: number of parameters does not match.'
    end if

    nstart = 1
    do n = 1, size(self % layers)
      nend = nstart + self % layers(n) % get_num_params() - 1
      if (nend - nstart < 1) cycle
      call self % layers(n) % set_params(params(nstart:nend))
      nstart = nend + 1
    end do

  end subroutine set_params


  module subroutine train(self, input_data, output_data, batch_size, &
                          epochs, optimizer)
    class(network), intent(in out) :: self
    real, intent(in) :: input_data(:,:)
    real, intent(in) :: output_data(:,:)
    integer, intent(in) :: batch_size
    integer, intent(in) :: epochs
    class(optimizer_base_type), intent(in) :: optimizer

    real :: pos
    integer :: dataset_size
    integer :: batch_start, batch_end
    integer :: i, j, n
    integer :: istart, iend, indices(2)

    dataset_size = size(output_data, dim=2)

    epoch_loop: do n = 1, epochs
      batch_loop: do i = 1, dataset_size / batch_size

        ! Pull a random mini-batch from the dataset
        call random_number(pos)
        batch_start = int(pos * (dataset_size - batch_size + 1)) + 1
        batch_end = batch_start + batch_size - 1

        ! FIXME shuffle in a way that doesn't require co_broadcast
        call co_broadcast(batch_start, 1)
        call co_broadcast(batch_end, 1)

        ! Distribute the batch in nearly equal pieces to all images
        indices = tile_indices(batch_size)
        istart = indices(1) + batch_start - 1
        iend = indices(2) + batch_start - 1

        do concurrent(j = istart:iend)
          call self % forward(input_data(:,j))
          call self % backward(output_data(:,j))
        end do

        select type (optimizer)
          type is (sgd)
            call self % update(optimizer % learning_rate / batch_size)
          class default
            error stop 'Unsupported optimizer'
        end select

      end do batch_loop
    end do epoch_loop

  end subroutine train


  module subroutine update(self, learning_rate)
    class(network), intent(in out) :: self
    real, intent(in) :: learning_rate
    call self % layers % update(learning_rate)
  end subroutine update

end submodule nf_network_submodule
