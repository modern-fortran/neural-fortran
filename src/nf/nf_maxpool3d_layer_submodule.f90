submodule(nf_maxpool3d_layer) nf_maxpool3d_layer_submodule

  implicit none

contains

  pure module function maxpool3d_layer_cons(pool_size, stride) result(res)
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(maxpool3d_layer) :: res
    res % pool_size = pool_size
    res % stride = stride
  end function maxpool3d_layer_cons


  module subroutine init(self, input_shape)
    class(maxpool3d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)
    self % width = input_shape(2) / self % stride
    self % height = input_shape(3) / self % stride
    self % depth = input_shape(4) / self % stride

    allocate(self % maxloc_x(self % channels, self % width, self % height, self % depth))
    self % maxloc_x = 0

    allocate(self % maxloc_y(self % channels, self % width, self % height, self % depth))
    self % maxloc_y = 0

    allocate(self % maxloc_z(self % channels, self % width, self % height, self % depth))
    self % maxloc_z = 0

    allocate(self % gradient(input_shape(1), input_shape(2), input_shape(3), input_shape(4)))
    self % gradient = 0

    allocate(self % output(self % channels, self % width, self % height, self % depth))
    self % output = 0

  end subroutine init


  pure module subroutine forward(self, input)
    class(maxpool3d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:,:)
    integer :: input_width, input_height, input_depth
    integer :: i, j, k, n
    integer :: ii, jj, kk
    integer :: iend, jend, kend
    integer :: iextent, jextent, kextent
    integer :: maxloc_xyz(3)

    input_width = size(input, dim=2)
    input_height = size(input, dim=3)
    input_depth = size(input, dim=4)

    iextent = input_width - mod(input_width, self % stride)
    jextent = input_height - mod(input_height, self % stride)
    kextent = input_depth - mod(input_depth, self % stride)

    ! Stride along the width, height and depth of the input volume
    stride_over_input: do concurrent( &
      i = 1:iextent:self % stride, &
      j = 1:jextent:self % stride, &
      k = 1:kextent:self % stride &
    )

      ! Indices of the pooling layer
      ii = i / self % stride + 1
      jj = j / self % stride + 1
      kk = k / self % stride + 1

      iend = i + self % pool_size - 1
      jend = j + self % pool_size - 1
      kend = k + self % pool_size - 1

      maxpool_for_each_channel: do concurrent(n = 1:self % channels)

        ! Get and store the location of the maximum value
        maxloc_xyz = maxloc(input(n,i:iend,j:jend,k:kend))
        self % maxloc_x(n,ii,jj,kk) = maxloc_xyz(1) + i - 1
        self % maxloc_y(n,ii,jj,kk) = maxloc_xyz(2) + j - 1
        self % maxloc_z(n,ii,jj,kk) = maxloc_xyz(3) + k - 1

        self % output(n,ii,jj,kk) = &
          input(n,self % maxloc_x(n,ii,jj,kk),self % maxloc_y(n,ii,jj,kk),self % maxloc_z(n,ii,jj,kk))

      end do maxpool_for_each_channel

    end do stride_over_input

  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    class(maxpool3d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:,:)
    real, intent(in) :: gradient(:,:,:,:)
    integer :: gradient_shape(4)
    integer :: channels, width, height, depth
    integer :: i, j, k, n

    gradient_shape = shape(gradient)
    channels = gradient_shape(1)
    width = gradient_shape(2)
    height = gradient_shape(3)
    depth = gradient_shape(4)

    ! The gradient of a max-pooling layer is just a value of the downstream
    ! gradient at the location of the maximum value, stored during the
    ! forward pass.
    do concurrent(n = 1:channels, i = 1:width, j = 1:height, k = 1:depth)
      associate(ii => self % maxloc_x(n,i,j,k), jj => self % maxloc_y(n,i,j,k), kk => self % maxloc_z(n,i,j,k))
        self % gradient(n,ii,jj,kk) = gradient(n,i,j,k)
      end associate
    end do

  end subroutine backward

end submodule nf_maxpool3d_layer_submodule
