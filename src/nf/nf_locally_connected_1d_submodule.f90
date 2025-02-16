submodule(nf_locally_connected_1d_layer) nf_locally_connected_1d_layer_submodule

  use nf_activation, only: activation_function
  use nf_random, only: random_normal

  implicit none

contains

  module function locally_connected_1d_layer_cons(filters, kernel_size, activation) result(res)
    implicit none
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_size
    class(activation_function), intent(in) :: activation
    type(locally_connected_1d_layer) :: res

    res % kernel_size = kernel_size
    res % filters = filters
    res % activation_name = activation % get_name()
    allocate( res % activation, source = activation )

  end function locally_connected_1d_layer_cons

  module subroutine init(self, input_shape)
    implicit none
    class(locally_connected_1d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)
    self % width = input_shape(2) - self % kernel_size + 1

    ! Output of shape filters x width
    allocate(self % output(self % filters, self % width))
    self % output = 0

    ! Kernel of shape filters x channels x kernel_size
    allocate(self % kernel(self % filters, self % channels, self % kernel_size))

    ! Initialize the kernel with random values with a normal distribution
    call random_normal(self % kernel)
    self % kernel = self % kernel / self % kernel_size ** 2

    allocate(self % biases(self % filters))
    self % biases = 0

    allocate(self % z, mold=self % output)
    self % z = 0

    allocate(self % gradient(input_shape(1), input_shape(2)))
    self % gradient = 0

    allocate(self % dw, mold=self % kernel)
    self % dw = 0

    allocate(self % db, mold=self % biases)
    self % db = 0

  end subroutine init

  pure module subroutine forward(self, input)
    implicit none
    class(locally_connected_1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    integer :: input_width, input_channels
    integer :: i, n, i_out
    integer :: iws, iwe
    integer :: half_window

    ! Get input dimensions
    input_channels = size(input, dim=1)
    input_width    = size(input, dim=2)

    ! For a kernel of odd size, half_window = kernel_size / 2 (integer division)
    half_window = self % kernel_size / 2

    ! Loop over output indices rather than input indices.
    do i_out = 1, self % width
      ! Compute the corresponding center index in the input.
      i = i_out + half_window

      ! Define the window in the input corresponding to the filter kernel
      iws = i - half_window
      iwe = i + half_window

      ! Compute the inner tensor product (sum of element-wise products)
      ! for each filter across all channels and positions in the kernel.
      do concurrent(n = 1:self % filters)
        self % z(n, i_out) = sum(self % kernel(n, :, :) * input(:, iws:iwe))
      end do

      ! Add the bias for each filter.
      self % z(:, i_out) = self % z(:, i_out) + self % biases
    end do

    ! Apply the activation function to get the final output.
    self % output = self % activation % eval(self % z)
  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    implicit none
    class(locally_connected_1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)     ! shape: (channels, width)
    real, intent(in) :: gradient(:,:)  ! shape: (filters, width)
    
    ! Local gradient arrays:
    real :: db(self % filters)
    real :: dw(self % filters, self % channels, self % kernel_size)
    real :: gdz(self % filters, size(input, 2))
    
    integer :: i, n, k
    integer :: input_channels, input_width
    integer :: istart, iend
    integer :: iws, iwe
    integer :: half_window
  
    ! Get input dimensions.
    input_channels = size(input, dim=1)
    input_width    = size(input, dim=2)
  
    ! For an odd-sized kernel, half_window = kernel_size / 2.
    half_window = self % kernel_size / 2
  
    ! Define the valid output range so that the full input window is available.
    istart = half_window + 1
    iend   = input_width - half_window
  
    !---------------------------------------------------------------------
    ! Compute the local gradient: gdz = (dL/dy) * sigma'(z)
    ! We assume self%z stores the pre-activation values from the forward pass.
    gdz = 0.0
    gdz(:, istart:iend) = gradient(:, istart:iend) * self % activation % eval_prime(self % z(:, istart:iend))
  
    !---------------------------------------------------------------------
    ! Compute gradient with respect to biases:
    ! dL/db(n) = sum_{i in valid range} gdz(n, i)
    do concurrent (n = 1:self % filters)
      db(n) = sum(gdz(n, istart:iend))
    end do
  
    ! Initialize weight gradient and input gradient accumulators.
    dw = 0.0
    self % gradient = 0.0  ! This array is assumed preallocated to shape (channels, width)
  
    !---------------------------------------------------------------------
    ! Accumulate gradients over valid output positions.
    ! For each output position i, determine the corresponding input window indices.
    do concurrent (n = 1:self % filters, &
                     k = 1:self % channels, &
                     i = istart:iend)
      ! The input window corresponding to output index i:
      iws = i - half_window
      iwe = i + half_window
  
      ! Weight gradient (dL/dw):
      ! For each kernel element, the contribution is the product of the input in the window
      ! and the local gradient at the output position i.
      dw(n, k, :) = dw(n, k, :) + input(k, iws:iwe) * gdz(n, i)
  
      ! Input gradient (dL/dx):
      ! Distribute the effect of the output gradient back onto the input window,
      ! weighted by the kernel weights.
      self % gradient(k, iws:iwe) = self % gradient(k, iws:iwe) + self % kernel(n, k, :) * gdz(n, i)
    end do
  
    !---------------------------------------------------------------------
    ! Accumulate the computed gradients into the layer's stored gradients.
    self % dw = self % dw + dw
    self % db = self % db + db
  
  end subroutine backward

  pure module function get_num_params(self) result(num_params)
    class(locally_connected_1d_layer), intent(in) :: self
    integer :: num_params
    num_params = product(shape(self % kernel)) + size(self % biases)
  end function get_num_params

  module function get_params(self) result(params)
    class(locally_connected_1d_layer), intent(in), target :: self
    real, allocatable :: params(:)
    real, pointer :: w_(:) => null()
    w_(1:size(self % kernel)) => self % kernel
    params = [ w_, self % biases ]
  end function get_params

  module function get_gradients(self) result(gradients)
    class(locally_connected_1d_layer), intent(in), target :: self
    real, allocatable :: gradients(:)
    real, pointer :: dw_(:) => null()
    dw_(1:size(self % dw)) => self % dw
    gradients = [ dw_, self % db ]
  end function get_gradients

  module subroutine set_params(self, params)
    class(locally_connected_1d_layer), intent(in out) :: self
    real, intent(in) :: params(:)

    if (size(params) /= self % get_num_params()) then
      error stop 'locally_connected_1d % set_params: Number of parameters does not match'
    end if

    self % kernel = reshape(params(:product(shape(self % kernel))), shape(self % kernel))

    associate(n => product(shape(self % kernel)))
      self % biases = params(n + 1 : n + self % filters)
    end associate

  end subroutine set_params

end submodule nf_locally_connected_1d_layer_submodule
