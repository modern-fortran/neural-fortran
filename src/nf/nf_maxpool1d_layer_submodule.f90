submodule(nf_maxpool1d_layer) nf_maxpool1d_layer_submodule

  implicit none

contains

  pure module function maxpool1d_layer_cons(pool_size, stride) result(res)
    implicit none
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(maxpool1d_layer) :: res
    res % pool_size = pool_size
    res % stride = stride
  end function maxpool1d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(maxpool1d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)
    self % width = input_shape(2) / self % stride

    allocate(self % maxloc(self % channels, self % width))
    self % maxloc = 0

    allocate(self % gradient(input_shape(1),input_shape(2)))
    self % gradient = 0

    allocate(self % output(self % channels, self % width))
    self % output = 0

  end subroutine init

  pure module subroutine forward(self, input)
    implicit none
    class(maxpool1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    integer :: input_width
    integer :: i, n
    integer :: ii
    integer :: iend
    integer :: iextent
    integer :: maxloc_x

    input_width = size(input, dim=2)

    iextent = input_width - mod(input_width, self % stride)

    ! Stride along the width of the input
    stride_over_input: do concurrent(i = 1:iextent:self % stride)
      
      ! Index of the pooling layer
      ii = i / self % stride + 1
      iend = i + self % pool_size - 1
      
      maxpool_for_each_channel: do concurrent(n = 1:self % channels)
        
        ! Get and store the location of the maximum value
        maxloc_x = maxloc(input(n, i:iend), dim=1)
        self % maxloc(n,ii) = maxloc_x + i - 1
        
        self % output(n,ii) = input(n, self % maxloc(n,ii))
        
      end do maxpool_for_each_channel
    
    end do stride_over_input
  
  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    implicit none
    class(maxpool1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    real, intent(in) :: gradient(:,:)
    integer :: gradient_shape(2)
    integer :: channels, width
    integer :: i, n

    gradient_shape = shape(gradient)
    channels = gradient_shape(1)
    width = gradient_shape(2)

    ! The gradient of a max-pooling layer is assigned to the stored max locations
    do concurrent(n = 1:channels, i = 1:width)
      self % gradient(n, self % maxloc(n,i)) = gradient(n,i)
    end do
  
  end subroutine backward


end submodule nf_maxpool1d_layer_submodule
