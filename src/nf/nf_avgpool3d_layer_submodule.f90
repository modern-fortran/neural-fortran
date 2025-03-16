submodule(nf_avgpool3d_layer) nf_avgpool3d_layer_submodule
  implicit none

contains

  pure module function avgpool3d_layer_cons(pool_size, stride) result(res)
    implicit none
    integer, intent(in) :: pool_size(3)
    integer, intent(in) :: stride(3)
    type(avgpool3d_layer) :: res

    res % pool_size = pool_size
    res % stride    = stride
  end function avgpool3d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(avgpool3d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
    ! input_shape is expected to be (channels, depth, height, width)

    self % channels = input_shape(1)
    self % depth    = input_shape(2) / self % stride(1)
    self % height   = input_shape(3) / self % stride(2)
    self % width    = input_shape(4) / self % stride(3)

    ! Allocate the gradient array corresponding to the input dimensions.
    allocate(self % gradient(input_shape(1), input_shape(2), input_shape(3), input_shape(4)))
    self % gradient = 0

    ! Allocate the output array (after pooling).
    allocate(self % output(self % channels, self % depth, self % height, self % width))
    self % output = 0
  end subroutine init


  pure module subroutine forward(self, input)
    implicit none
    class(avgpool3d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:,:)
    integer :: input_depth, input_height, input_width
    integer :: i, j, k, n
    integer :: ii, jj, kk, iend, jend, kend
    integer :: kdepth, jheight, kwidth

    input_depth   = size(input, dim=2)
    input_height  = size(input, dim=3)
    input_width   = size(input, dim=4)
    
    ! Ensure we only process complete pooling regions.
    kdepth = input_depth - mod(input_depth, self % stride(1))
    jheight = input_height - mod(input_height, self % stride(2))
    kwidth = input_width - mod(input_width, self % stride(3))

    ! Loop over the input with a step size equal to the stride and over all channels.
    do concurrent (i = 1:kdepth:self % stride(1), j = 1:jheight:self % stride(2), k = 1:kwidth:self % stride(3), n = 1:self % channels)
      ii = (i - 1) / self % stride(1) + 1
      jj = (j - 1) / self % stride(2) + 1
      kk = (k - 1) / self % stride(3) + 1
      
      iend = min(i + self % pool_size(1) - 1, input_depth)
      jend = min(j + self % pool_size(2) - 1, input_height)
      kend = min(k + self % pool_size(3) - 1, input_width)
      
      ! Compute the average over the pooling region.
      self % output(n, ii, jj, kk) = sum(input(n, i:iend, j:jend, k:kend)) / ((iend - i + 1) * (jend - j + 1) * (kend - k + 1))
    end do
  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    implicit none
    class(avgpool3d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:,:)
    real, intent(in) :: gradient(:,:,:,:)
    integer :: channels, pooled_depth, pooled_height, pooled_width
    integer :: i, j, k, n, x, y, z, istart, iend, jstart, jend, kstart, kend
    real :: scale_factor

    channels      = size(gradient, dim=1)
    pooled_depth  = size(gradient, dim=2)
    pooled_height = size(gradient, dim=3)
    pooled_width  = size(gradient, dim=4)

    ! The gradient for average pooling is distributed evenly over the pooling window.
    do concurrent (n = 1:channels, i = 1:pooled_depth, j = 1:pooled_height, k = 1:pooled_width)
      istart = (i - 1) * self % stride(1) + 1
      iend   = min(istart + self % pool_size(1) - 1, size(input, dim=2))
      jstart = (j - 1) * self % stride(2) + 1
      jend   = min(jstart + self % pool_size(2) - 1, size(input, dim=3))
      kstart = (k - 1) * self % stride(3) + 1
      kend   = min(kstart + self % pool_size(3) - 1, size(input, dim=4))
      scale_factor = 1.0 / ((iend - istart + 1) * (jend - jstart + 1) * (kend - kstart + 1))

      do concurrent (x = istart:iend, y = jstart:jend, z = kstart:kend)
        self % gradient(n, x, y, z) = gradient(n, i, j, k) * scale_factor
      end do
    end do
  end subroutine backward

end submodule nf_avgpool3d_layer_submodule
