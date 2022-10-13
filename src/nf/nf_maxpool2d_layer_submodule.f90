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

    allocate(self % gradient(input_shape(1),input_shape(2),input_shape(3)))
    self % gradient = 0

    allocate(self % output(self % channels, self % width, self % height))
    self % output = 0

  end subroutine init


  pure module subroutine forward(self, input)
    implicit none
    class(maxpool2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    integer :: input_width, input_height
    integer :: i, j, n
    integer :: ii, jj
    integer :: iend, jend
    integer :: iextent, jextent
    integer :: maxloc_xy(2)

    input_width = size(input, dim=2)
    input_height = size(input, dim=3)

    iextent = input_width - mod(input_width, self % stride)
    jextent = input_height - mod(input_height, self % stride)

    ! Stride along the width and height of the input image
    stride_over_input: do concurrent( &
      i = 1:iextent:self % stride, &
      j = 1:jextent:self % stride &
    )

      ! Indices of the pooling layer
      ii = i / self % stride + 1
      jj = j / self % stride + 1

      iend = i + self % pool_size - 1
      jend = j + self % pool_size - 1

      maxpool_for_each_channel: do concurrent(n = 1:self % channels)

        ! Get and store the location of the maximum value
        maxloc_xy = maxloc(input(n,i:iend,j:jend))
        self % maxloc_x(n,ii,jj) = maxloc_xy(1) + i - 1
        self % maxloc_y(n,ii,jj) = maxloc_xy(2) + j - 1

        self % output(n,ii,jj) = &
          input(n,self % maxloc_x(n,ii,jj),self % maxloc_y(n,ii,jj))

      end do maxpool_for_each_channel

    end do stride_over_input

  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    implicit none
    class(maxpool2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:,:,:)
    integer :: gradient_shape(3)
    integer :: channels, width, height
    integer :: i, j, n

    gradient_shape = shape(gradient)
    channels = gradient_shape(1)
    width = gradient_shape(2)
    height = gradient_shape(3)

    ! The gradient of a max-pooling layer is just a value of the downstream
    ! gradient at the location of the maximum value, stored during the
    ! forward pass.
    do concurrent(n = 1:channels, i = 1:width, j = 1:height)
      associate(ii => self % maxloc_x(n,i,j), jj => self % maxloc_y(n,i,j))
        self % gradient(n,ii,jj) = gradient(n,i,j)
      end associate
    end do

  end subroutine backward

end submodule nf_maxpool2d_layer_submodule
