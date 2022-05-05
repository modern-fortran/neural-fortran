submodule(nf_conv2d_layer) nf_conv2d_layer_submodule

  implicit none

contains

  pure module function conv2d_layer_cons(window_size, filters, activation) result(res)
    integer, intent(in) :: window_size
    integer, intent(in) :: filters
    character(*), intent(in) :: activation
    type(conv2d_layer) :: res
    res % window_size = window_size
    res % filters = filters
    call res % set_activation(activation)
  end function conv2d_layer_cons


  module subroutine init(self, input_shape)
    class(conv2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % width = input_shape(1) - self % window_size + 1
    self % height = input_shape(2) - self % window_size + 1
    self % channels = input_shape(3)

    allocate(self % output(self % width, self % height, self % filters))
    self % output = 0

    allocate(self % kernel(self % window_size, self % window_size, &
                           self % channels, self % filters))
    self % kernel = 0 ! TODO 4-d randn

    allocate(self % biases(self % filters))
    self % biases = 0

  end subroutine init


  module subroutine forward(self, input)
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    integer :: i, j, n

    do j = 2, self % height - 1 ! FIXME
      do i = 2, self % width - 1 ! FIXME

        do n = 1, self % filters
          !FIXME indices of self % output
          self % output(n,i,j) = &
            sum(self % kernel(n,:,i-1:i+1,j-1:j+1) * input(:,i-1:i+1,j-1:j+1)) &
            + self % biases(n)
        end do

      end do
    end do

  end subroutine forward


  module subroutine backward(self, input, gradient)
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:,:,:)
    print *, 'Warning: conv2d backward pass not implemented'
  end subroutine backward

end submodule nf_conv2d_layer_submodule
