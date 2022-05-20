submodule(nf_network) nf_network_submodule

  use nf_dense_layer, only: dense_layer
  use nf_input1d_layer, only: input1d_layer
  use nf_input3d_layer, only: input3d_layer
  use nf_layer, only: layer
  use nf_loss, only: quadratic_derivative
  use nf_optimizers, only: sgd
  use nf_parallel, only: tile_indices

  implicit none

contains

  module function network_cons(layers) result(res)
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

    res % layers = layers

    ! Loop over each layer in order and call the init methods.
    ! This will allocate the data internal to each layer (e.g. weights, biases)
    ! according to the size of the previous layer.
    do n = 2, size(layers)
      call res % layers(n) % init(res % layers(n - 1))
    end do

  end function network_cons


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


  module function output(self, input) result(res)
    class(network), intent(in out) :: self
    real, intent(in) :: input(:)
    real, allocatable :: res(:)
    integer :: num_layers

    num_layers = size(self % layers)

    call self % forward(input)

    select type(output_layer => self % layers(num_layers) % p); type is(dense_layer)
      res = output_layer % output
    end select

  end function output


  module subroutine print_info(self)
    class(network), intent(in) :: self
    call self % layers % print_info()
  end subroutine print_info


  module subroutine train(self, input_data, output_data, batch_size, &
                          epochs, optimizer)
    class(network), intent(in out) :: self
    real, intent(in) :: input_data(:,:)
    real, intent(in) :: output_data(:,:)
    integer, intent(in) :: batch_size
    integer, intent(in) :: epochs
    type(sgd), intent(in) :: optimizer

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

      call self % update(optimizer % learning_rate / batch_size)

      end do batch_loop
    end do epoch_loop

  end subroutine train


  module subroutine update(self, learning_rate)
    class(network), intent(in out) :: self
    real, intent(in) :: learning_rate
    call self % layers % update(learning_rate)
  end subroutine update

end submodule nf_network_submodule
