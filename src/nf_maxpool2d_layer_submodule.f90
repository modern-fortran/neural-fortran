submodule(nf_maxpool2d_layer) nf_maxpool2d_layer_submodule

  implicit none

contains

  pure module function maxpool2d_layer_cons(pool_size, stride) result(res)
    implicit none
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(maxpool2d_layer) :: res
    res % pool_size = pool_size
    res % stride = stride
  end function maxpool2d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(maxpool2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)
    self % width = input_shape(2) / self % stride
    self % height = input_shape(3) / self % stride

    allocate(self % maxloc_x(self % channels, self % width, self % height))
    self % maxloc_x = 0

    allocate(self % maxloc_y(self % channels, self % width, self % height))
    self % maxloc_y = 0

    allocate(self % output(self % channels, self % width, self % height))
    self % output = 0

  end subroutine init


  module subroutine forward(self, input)
    implicit none
    class(maxpool2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    integer :: input_width, input_height
    integer :: i, j, n
    integer :: ii, jj
    integer :: iend, jend

    input_width = size(input, dim=2)
    input_height = size(input, dim=2)

    ! Stride along the width and height of the input image
    do concurrent( &
      i = 1:input_width:self % stride, &
      j = 1:input_height:self % stride &
    )

      ! Indices of the pooling layer
      ii = i / self % stride + 1
      jj = j / self % stride + 1

      iend = i + self % pool_size - 1
      jend = j + self % pool_size - 1

      do concurrent(n = 1:self % channels)
        !TODO find and store maxloc
        self % output(n,ii,jj) = maxval(input(n,i:iend,j:jend))
      end do

    end do

  end subroutine forward


  module subroutine backward(self, input, gradient)
    implicit none
    class(maxpool2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:,:,:)
    print *, 'Warning: maxpool2d backward pass not implemented'
  end subroutine backward

end submodule nf_maxpool2d_layer_submodule
