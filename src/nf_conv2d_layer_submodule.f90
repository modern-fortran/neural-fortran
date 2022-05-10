submodule(nf_conv2d_layer) nf_conv2d_layer_submodule

  use nf_random, only: randn

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

    ! Initialize the kernel with random values with a normal distribution.
    self % kernel = randn(self % window_size, self % window_size, &
                          self % channels, self % filters) &
                  / self % window_size**2 !TODO window_width * window_height

    allocate(self % biases(self % filters))
    self % biases = 0

  end subroutine init


  pure module subroutine forward(self, input)
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    integer :: input_width, input_height, input_channels
    integer :: istart, iend
    integer :: jstart, jend
    integer :: i, j, n
    integer :: ii, jj

    input_width = size(input, dim=1)
    input_height = size(input, dim=2)
    input_channels = size(input, dim=3)

    istart = self % window_size / 2 + 1 ! TODO window_width
    jstart = self % window_size / 2 + 1 ! TODO window_height
    iend = input_width - istart + 1
    jend = input_height - jstart + 1

    do j = jstart, jend
      do i = istart, iend

        ! Indices of the output matrix
        ii = i - self % window_size / 2 ! TODO window_width
        jj = j - self % window_size / 2 ! TODO window_height

        do n = 1, self % filters

          associate( &
            kernel => self % kernel(:,:,:,n), &
            filtered_input => input(i-1:i+1,j-1:j+1,:) &
          )

            self % output(ii,jj,n) = sum(kernel * filtered_input) &
                                 + self % biases(n)

          end associate

        end do

        self % output(ii,jj,:) = self % activation(self % output(ii,jj,:))

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
