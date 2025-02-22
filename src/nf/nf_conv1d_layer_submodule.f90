submodule(nf_conv1d_layer) nf_conv1d_layer_submodule

  use nf_activation, only: activation_function
  use nf_random, only: random_normal

  implicit none

contains

  module function conv1d_layer_cons(filters, kernel_size, activation) result(res)
    implicit none
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_size
    class(activation_function), intent(in) :: activation
    type(conv1d_layer) :: res

    res % kernel_size = kernel_size
    res % filters = filters
    res % activation_name = activation % get_name()
    allocate( res % activation, source = activation )
  end function conv1d_layer_cons

  module subroutine init(self, input_shape)
    implicit none
    class(conv1d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)
    self % width = input_shape(2) - self % kernel_size + 1

    ! Output of shape: filters x width
    allocate(self % output(self % filters, self % width))
    self % output = 0

    ! Kernel of shape: filters x channels x kernel_size
    allocate(self % kernel(self % filters, self % channels, self % kernel_size))
    call random_normal(self % kernel)
    self % kernel = self % kernel / real(self % kernel_size**2)

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
    class(conv1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    integer :: input_channels, input_width
    integer :: j, n
    integer :: iws, iwe, half_window

    input_channels = size(input, dim=1)
    input_width    = size(input, dim=2)
    half_window = self % kernel_size / 2

    ! Loop over output positions.
    do j = 1, self % width
      ! Compute the input window corresponding to output index j.
      ! In forward: center index = j + half_window, so window = indices j to j+kernel_size-1.
      iws = j
      iwe = j + self % kernel_size - 1

      ! For each filter, compute the convolution (inner product over channels and kernel width).
      do concurrent (n = 1:self % filters)
        self % z(n, j) = sum(self % kernel(n, :, :) * input(:, iws:iwe))
      end do

      ! Add the bias for each filter.
      self % z(:, j) = self % z(:, j) + self % biases
    end do

    ! Apply the activation function.
    self % output = self % activation % eval(self % z)
  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    implicit none
    class(conv1d_layer), intent(in out) :: self
    ! 'input' has shape: (channels, input_width)
    ! 'gradient' (dL/dy) has shape: (filters, output_width)
    real, intent(in) :: input(:,:)
    real, intent(in) :: gradient(:,:)

    integer :: input_channels, input_width, output_width
    integer :: j, n, k
    integer :: iws, iwe, half_window
    real :: gdz_val

    ! Local arrays to accumulate gradients.
    real :: gdz(self % filters, self % width)  ! local gradient (dL/dz)
    real :: db_local(self % filters)
    real :: dw_local(self % filters, self % channels, self % kernel_size)

    ! Determine dimensions.
    input_channels = size(input, dim=1)
    input_width    = size(input, dim=2)
    output_width   = self % width    ! Note: output_width = input_width - kernel_size + 1

    half_window = self % kernel_size / 2

    !--- Compute the local gradient gdz = (dL/dy) * sigma'(z) for each output.
    do j = 1, output_width
       gdz(:, j) = gradient(:, j) * self % activation % eval_prime(self % z(:, j))
    end do

    !--- Compute bias gradients: db(n) = sum_j gdz(n, j)
    do n = 1, self % filters
       db_local(n) = sum(gdz(n, :))
    end do

    !--- Initialize weight gradient and input gradient accumulators.
    dw_local = 0.0
    self % gradient = 0.0

    !--- Accumulate gradients over each output position.
    ! In the forward pass the window for output index j was:
    !   iws = j,  iwe = j + kernel_size - 1.
    do n = 1, self % filters
       do j = 1, output_width
          iws = j
          iwe = j + self % kernel_size - 1
          do k = 1, self % channels
             ! Weight gradient: accumulate contribution from the input window.
             dw_local(n, k, :) = dw_local(n, k, :) + input(k, iws:iwe) * gdz(n, j)
             ! Input gradient: propagate gradient back to the input window.
             self % gradient(k, iws:iwe) = self % gradient(k, iws:iwe) + self % kernel(n, k, :) * gdz(n, j)
          end do
       end do
    end do

    !--- Update stored gradients.
    self % dw = self % dw + dw_local
    self % db = self % db + db_local

  end subroutine backward

  pure module function get_num_params(self) result(num_params)
    class(conv1d_layer), intent(in) :: self
    integer :: num_params
    num_params = product(shape(self % kernel)) + size(self % biases)
  end function get_num_params

  module function get_params(self) result(params)
    class(conv1d_layer), intent(in), target :: self
    real, allocatable :: params(:)
    real, pointer :: w_(:) => null()
    w_(1:size(self % kernel)) => self % kernel
    params = [ w_, self % biases ]
  end function get_params

  module function get_gradients(self) result(gradients)
    class(conv1d_layer), intent(in), target :: self
    real, allocatable :: gradients(:)
    real, pointer :: dw_(:) => null()
    dw_(1:size(self % dw)) => self % dw
    gradients = [ dw_, self % db ]
  end function get_gradients

  module subroutine set_params(self, params)
    class(conv1d_layer), intent(in out) :: self
    real, intent(in) :: params(:)

    if (size(params) /= self % get_num_params()) then
      error stop 'conv1d_layer % set_params: Number of parameters does not match'
    end if

    self % kernel = reshape(params(:product(shape(self % kernel))), shape(self % kernel))
    associate(n => product(shape(self % kernel)))
      self % biases = params(n + 1 : n + self % filters)
    end associate

  end subroutine set_params

end submodule nf_conv1d_layer_submodule
