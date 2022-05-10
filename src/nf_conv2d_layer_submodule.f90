submodule(nf_conv2d_layer) nf_conv2d_layer_submodule

  use nf_random, only: randn

  implicit none

contains

  pure module function conv2d_layer_cons(window_size, filters, activation) result(res)
    implicit none
    integer, intent(in) :: window_size
    integer, intent(in) :: filters
    character(*), intent(in) :: activation
    type(conv2d_layer) :: res
    res % window_size = window_size
    res % filters = filters
    call res % set_activation(activation)
  end function conv2d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(conv2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)
    self % width = input_shape(2) - self % window_size + 1
    self % height = input_shape(3) - self % window_size + 1

    ! Output of shape filters x width x height
    allocate(self % output(self % filters, self % width, self % height))
    self % output = 0

    ! Kernel of shape filters x channels x width x height
    allocate(self % kernel(self % filters, self % channels, &
                           self % window_size, self % window_size))

    ! Initialize the kernel with random values with a normal distribution.
    self % kernel = randn(self % filters, self % channels, &
                          self % window_size, self % window_size) &
                  / self % window_size**2 !TODO window_width * window_height

    allocate(self % biases(self % filters))
    self % biases = 0

  end subroutine init


  pure module subroutine forward(self, input)
    implicit none
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    integer :: input_width, input_height, input_channels
    integer :: istart, iend
    integer :: jstart, jend
    integer :: i, j, n
    integer :: ii, jj

    ! Input dimensions are channels x width x height
    input_channels = size(input, dim=1)
    input_width = size(input, dim=2)
    input_height = size(input, dim=3)

    ! Determine the start and end indices for the width and height dimensions
    ! of the input that correspond to the center of each filter (window).
    istart = self % window_size / 2 + 1 ! TODO window_width
    jstart = self % window_size / 2 + 1 ! TODO window_height
    iend = input_width - istart + 1
    jend = input_height - jstart + 1

    convolution: do concurrent(i = istart:iend, j = jstart:jend)

      ! Indices of the output matrix
      ii = i - self % window_size / 2 ! TODO window_width
      jj = j - self % window_size / 2 ! TODO window_height

      inner_product: do concurrent(n = 1:self % filters)
        self % output(n,ii,jj) = &
          sum(self % kernel(n,:,:,:) * input(:,i-1:i+1,j-1:j+1)) &
          + self % biases(n)
      end do inner_product

      ! Activate
      self % output(:,ii,jj) = self % activation(self % output(:,ii,jj))

    end do convolution

  end subroutine forward


  module subroutine backward(self, input, gradient)
    implicit none
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:,:,:)
    print *, 'Warning: conv2d backward pass not implemented'
  end subroutine backward

end submodule nf_conv2d_layer_submodule
