submodule(nf_maxpool1d_layer) nf_maxpool1d_layer_submodule
  implicit none

contains

  pure module function maxpool1d_layer_cons(pool_size, stride) result(res)
    implicit none
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(maxpool1d_layer) :: res

    res % pool_size = pool_size
    res % stride    = stride
  end function maxpool1d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(maxpool1d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
    ! input_shape is expected to be (channels, width)

    self % channels = input_shape(1)
    ! The new width is the integer division of the input width by the stride.
    self % width    = input_shape(2) / self % stride

    ! Allocate storage for the index of the maximum element within each pooling region.
    allocate(self % maxloc(self % channels, self % width))
    self % maxloc = 0

    ! Allocate the gradient array corresponding to the input dimensions.
    allocate(self % gradient(input_shape(1), input_shape(2)))
    self % gradient = 0

    ! Allocate the output array (after pooling).
    allocate(self % output(self % channels, self % width))
    self % output = 0
  end subroutine init


  pure module subroutine forward(self, input)
    implicit none
    class(maxpool1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    integer :: input_width
    integer :: i, n
    integer :: ii, iend
    integer :: iextent
    integer :: max_index  ! Temporary variable to hold the local index of the max
    integer :: maxloc_temp(1)  ! Temporary array to hold the result of maxloc

    input_width = size(input, dim=2)
    ! Ensure we only process complete pooling regions.
    iextent = input_width - mod(input_width, self % stride)

    ! Loop over the input with a step size equal to the stride and over all channels.
    do concurrent (i = 1:iextent: self % stride, n = 1:self % channels)
      ! Compute the index in the pooled (output) array.
      ii = (i - 1) / self % stride + 1
      ! Determine the ending index of the current pooling region.
      iend = min(i + self % pool_size - 1, input_width)

      ! Find the index (within the pooling window) of the maximum value.
      maxloc_temp = maxloc(input(n, i:iend))
      max_index = maxloc_temp(1) + i - 1  ! Adjust to the index in the original input

      ! Store the location of the maximum value.
      self % maxloc(n, ii) = max_index
      ! Set the output as the maximum value from this pooling region.
      self % output(n, ii) = input(n, max_index)
    end do
  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    implicit none
    class(maxpool1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    real, intent(in) :: gradient(:,:)
    integer :: channels, pooled_width
    integer :: i, n

    channels    = size(gradient, dim=1)
    pooled_width = size(gradient, dim=2)

    ! The gradient for max-pooling is nonzero only at the input locations
    ! that were the maxima during the forward pass.
    do concurrent (n = 1:channels, i = 1:pooled_width)
      self % gradient(n, self % maxloc(n, i)) = gradient(n, i)
    end do
  end subroutine backward

end submodule nf_maxpool1d_layer_submodule