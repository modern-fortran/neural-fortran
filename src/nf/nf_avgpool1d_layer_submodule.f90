submodule(nf_avgpool1d_layer) nf_avgpool1d_layer_submodule
  implicit none

contains

  pure module function avgpool1d_layer_cons(pool_width, stride) result(res)
    implicit none
    integer, intent(in) :: pool_width
    integer, intent(in) :: stride
    type(avgpool1d_layer) :: res

    res % pool_width = pool_width
    res % stride    = stride
  end function avgpool1d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(avgpool1d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
    ! input_shape is expected to be (channels, width)

    self % channels = input_shape(1)
    ! The new width is the integer division of the input width by the stride.
    self % width    = input_shape(2) / self % stride

    ! Allocate the gradient array corresponding to the input dimensions.
    allocate(self % gradient(input_shape(1), input_shape(2)))
    self % gradient = 0

    ! Allocate the output array (after pooling).
    allocate(self % output(self % channels, self % width))
    self % output = 0
  end subroutine init


  pure module subroutine forward(self, input)
    implicit none
    class(avgpool1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    integer :: input_width
    integer :: i, n
    integer :: ii, iend
    integer :: iextent

    input_width = size(input, dim=2)
    ! Ensure we only process complete pooling regions.
    iextent = input_width - mod(input_width, self % stride)

    ! Loop over the input with a step size equal to the stride and over all channels.
    do concurrent (i = 1:iextent: self % stride, n = 1:self % channels)
      ! Compute the index in the pooled (output) array.
      ii = (i - 1) / self % stride + 1
      ! Determine the ending index of the current pooling region.
      iend = min(i + self % pool_width - 1, input_width)

      ! Compute the average over the pooling region.
      self % output(n, ii) = sum(input(n, i:iend)) / (iend - i + 1)
    end do
  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    implicit none
    class(avgpool1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    real, intent(in) :: gradient(:,:)
    integer :: channels, pooled_width
    integer :: i, n, j, istart, iend
    real :: scale_factor

    channels    = size(gradient, dim=1)
    pooled_width = size(gradient, dim=2)

    ! The gradient for average pooling is distributed evenly over the pooling window.
    do concurrent (n = 1:channels, i = 1:pooled_width)
      istart = (i - 1) * self % stride + 1
      iend = min(istart + self % pool_width - 1, size(input, dim=2))
      scale_factor = 1.0 / (iend - istart + 1)

      do j = istart, iend
        self % gradient(n, j) = gradient(n, i) * scale_factor
      end do
    end do
  end subroutine backward

end submodule nf_avgpool1d_layer_submodule