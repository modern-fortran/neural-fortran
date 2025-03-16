submodule(nf_maxpool3d_layer) nf_maxpool3d_layer_submodule

  implicit none

contains

  pure module function maxpool3d_layer_cons(pool_size, stride) result(res)
    implicit none
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(maxpool3d_layer) :: res
    res % pool_size = pool_size
    res % stride = stride
  end function maxpool3d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(maxpool3d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)
    self % depth = input_shape(2) / self % stride
    self % width = input_shape(3) / self % stride
    self % height = input_shape(4) / self % stride

    allocate(self % maxloc_x(self % channels, self % depth, self % width, self % height))
    allocate(self % maxloc_y(self % channels, self % depth, self % width, self % height))
    allocate(self % maxloc_z(self % channels, self % depth, self % width, self % height))
    self % maxloc_x = 0
    self % maxloc_y = 0
    self % maxloc_z = 0

    allocate(self % gradient(input_shape(1), input_shape(2), input_shape(3), input_shape(4)))
    self % gradient = 0

    allocate(self % output(self % channels, self % depth, self % width, self % height))
    self % output = 0

  end subroutine init


  pure module subroutine forward(self, input)
    implicit none
    class(maxpool3d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:,:)
    integer :: input_depth, input_width, input_height
    integer :: i, j, k, n
    integer :: ii, jj, kk
    integer :: iend, jend, kend
    integer :: iextent, jextent, kextent
    integer :: maxloc_xyz(3)

    input_depth = size(input, dim=2)
    input_width = size(input, dim=3)
    input_height = size(input, dim=4)

    kextent = input_depth - mod(input_depth, self % stride)
    iextent = input_width - mod(input_width, self % stride)
    jextent = input_height - mod(input_height, self % stride)

    do concurrent( &
      k = 1:kextent:self % stride, &
      i = 1:iextent:self % stride, &
      j = 1:jextent:self % stride &
    )

      kk = k / self % stride + 1
      ii = i / self % stride + 1
      jj = j / self % stride + 1

      kend = k + self % pool_size - 1
      iend = i + self % pool_size - 1
      jend = j + self % pool_size - 1

      do concurrent(n = 1:self % channels)
        maxloc_xyz = maxloc(input(n, k:kend, i:iend, j:jend))
        self % maxloc_x(n,kk,ii,jj) = maxloc_xyz(1) + k - 1
        self % maxloc_y(n,kk,ii,jj) = maxloc_xyz(2) + i - 1
        self % maxloc_z(n,kk,ii,jj) = maxloc_xyz(3) + j - 1

        self % output(n,kk,ii,jj) = &
          input(n,self % maxloc_x(n,kk,ii,jj),self % maxloc_y(n,kk,ii,jj),self % maxloc_z(n,kk,ii,jj))
      end do

    end do

  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    implicit none
    class(maxpool3d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:,:)
    real, intent(in) :: gradient(:,:,:,:)
    integer :: gradient_shape(4)
    integer :: channels, depth, width, height
    integer :: i, j, k, n

    gradient_shape = shape(gradient)
    channels = gradient_shape(1)
    depth = gradient_shape(2)
    width = gradient_shape(3)
    height = gradient_shape(4)

    do concurrent(n = 1:channels, k = 1:depth, i = 1:width, j = 1:height)
      associate(ii => self % maxloc_x(n,k,i,j), jj => self % maxloc_y(n,k,i,j), kk => self % maxloc_z(n,k,i,j))
        self % gradient(n,ii,jj,kk) = gradient(n,k,i,j)
      end associate
    end do

  end subroutine backward

end submodule nf_maxpool3d_layer_submodule