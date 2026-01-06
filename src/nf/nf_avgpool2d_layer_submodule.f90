submodule(nf_avgpool2d_layer) nf_avgpool2d_layer_submodule
  implicit none

contains

  pure module function avgpool2d_layer_cons(pool_width, pool_height, stride) result(res)
    implicit none
    integer, intent(in) :: pool_width
    integer, intent(in) :: pool_height
    integer, intent(in) :: stride
    type(avgpool2d_layer) :: res

    res % pool_width = pool_width
    res % pool_height = pool_height
    res % stride = stride
  end function avgpool2d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(avgpool2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
    ! input_shape is expected to be (channels, width, height)

    self % channels = input_shape(1)
    self % width = input_shape(2) / self % stride
    self % height = input_shape(3) / self % stride

    ! Allocate the gradient array corresponding to the input dimensions.
    allocate(self % gradient(input_shape(1), input_shape(2), input_shape(3)))
    self % gradient = 0

    ! Allocate the output array (after pooling).
    allocate(self % output(self % channels, self % width, self % height))
    self % output = 0
  end subroutine init


  pure module subroutine forward(self, input)
    implicit none
    class(avgpool2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    integer :: input_width, input_height
    integer :: i, j, n
    integer :: ii, jj, iend, jend
    integer :: iextent, jextent

    input_width  = size(input, dim=2)
    input_height = size(input, dim=3)
    
    ! Ensure we only process complete pooling regions.
    iextent = input_width - mod(input_width, self % stride)
    jextent = input_height - mod(input_height, self % stride)

    ! Loop over the input with a step size equal to the stride and over all channels.
    do concurrent (i = 1:iextent:self % stride, j = 1:jextent:self % stride, n = 1:self % channels)
      ii = (i - 1) / self % stride + 1
      jj = (j - 1) / self % stride + 1
      
      iend = min(i + self % pool_width - 1, input_width)
      jend = min(j + self % pool_height - 1, input_height)
      
      ! Compute the average over the pooling region.
      self % output(n, ii, jj) = sum(input(n, i:iend, j:jend)) / ((iend - i + 1) * (jend - j + 1))
    end do
  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    implicit none
    class(avgpool2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:,:,:)
    integer :: channels, pooled_width, pooled_height
    integer :: i, j, n, x, y, istart, iend, jstart, jend
    real :: scale_factor

    channels      = size(gradient, dim=1)
    pooled_width  = size(gradient, dim=2)
    pooled_height = size(gradient, dim=3)

    ! The gradient for average pooling is distributed evenly over the pooling window.
    do concurrent (n = 1:channels, i = 1:pooled_width, j = 1:pooled_height)
      istart = (i - 1) * self % stride + 1
      iend   = min(istart + self % pool_width - 1, size(input, dim=2))
      jstart = (j - 1) * self % stride + 1
      jend   = min(jstart + self % pool_height - 1, size(input, dim=3))
      scale_factor = 1.0 / ((iend - istart + 1) * (jend - jstart + 1))

      do concurrent (x = istart:iend, y = jstart:jend)
        self % gradient(n, x, y) = gradient(n, i, j) * scale_factor
      end do
    end do
  end subroutine backward

end submodule nf_avgpool2d_layer_submodule
