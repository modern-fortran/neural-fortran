submodule(nf_locally_connected1d_layer) nf_locally_connected1d_layer_submodule

  use nf_activation, only: activation_function
  use nf_random, only: random_normal

  implicit none

contains

  module function locally_connected1d_layer_cons(filters, kernel_size, activation) result(res)
    implicit none
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_size
    class(activation_function), intent(in) :: activation
    type(locally_connected1d_layer) :: res

    res % kernel_size = kernel_size
    res % filters = filters
    res % activation_name = activation % get_name()
    allocate(res % activation, source = activation)
  end function locally_connected1d_layer_cons

  module subroutine init(self, input_shape)
    implicit none
    class(locally_connected1d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)
    self % width = input_shape(2) - self % kernel_size + 1

    allocate(self % output(self % filters, self % width))
    self % output = 0

    allocate(self % kernel(self % filters, self % width, self % channels, self % kernel_size))
    call random_normal(self % kernel)
    self % kernel = self % kernel / real(self % kernel_size**2)

    allocate(self % biases(self % filters, self % width))
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
    class(locally_connected1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    integer :: input_channels, input_width
    integer :: j, n
    integer :: iws, iwe

    input_channels = size(input, dim=1)
    input_width    = size(input, dim=2)

    do j = 1, self % width
      iws = j
      iwe = j + self % kernel_size - 1
      do n = 1, self % filters
        self % z(n, j) = sum(self % kernel(n, j, :, :) * input(:, iws:iwe)) + self % biases(n, j)
      end do
    end do
    self % output = self % activation % eval(self % z)
  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    implicit none
    class(locally_connected1d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    real, intent(in) :: gradient(:,:)
    integer :: input_channels, input_width, output_width
    integer :: j, n, k
    integer :: iws, iwe
    real :: gdz(self % filters, self % width)
    real :: db_local(self % filters, self % width)
    real :: dw_local(self % filters, self % width, self % channels, self % kernel_size)

    input_channels = size(input, dim=1)
    input_width    = size(input, dim=2)
    output_width   = self % width

    do j = 1, output_width
       gdz(:, j) = gradient(:, j) * self % activation % eval_prime(self % z(:, j))
    end do

    do n = 1, self % filters
       do j = 1, output_width
          db_local(n, j) = gdz(n, j)
       end do
    end do

    dw_local = 0.0
    self % gradient = 0.0

    do n = 1, self % filters
       do j = 1, output_width
          iws = j
          iwe = j + self % kernel_size - 1
          do k = 1, self % channels
             dw_local(n, j, k, :) = dw_local(n, j, k, :) + input(k, iws:iwe) * gdz(n, j)
             self % gradient(k, iws:iwe) = self % gradient(k, iws:iwe) + self % kernel(n, j, k, :) * gdz(n, j)
          end do
       end do
    end do

    self % dw = self % dw + dw_local
    self % db = self % db + db_local
  end subroutine backward

  pure module function get_num_params(self) result(num_params)
    class(locally_connected1d_layer), intent(in) :: self
    integer :: num_params
    num_params = product(shape(self % kernel)) + product(shape(self % biases))
  end function get_num_params

  module function get_params(self) result(params)
    class(locally_connected1d_layer), intent(in), target :: self
    real, allocatable :: params(:)
    params = [self % kernel, self % biases]
  end function get_params

  module function get_gradients(self) result(gradients)
    class(locally_connected1d_layer), intent(in), target :: self
    real, allocatable :: gradients(:)
    gradients = [self % dw, self % db]
  end function get_gradients

  module subroutine set_params(self, params)
    class(locally_connected1d_layer), intent(in out) :: self
    real, intent(in) :: params(:)

    if (size(params) /= self % get_num_params()) then
      error stop 'locally_connected1d_layer % set_params: Number of parameters does not match'
    end if

    self % kernel = reshape(params(:product(shape(self % kernel))), shape(self % kernel))
    associate(n => product(shape(self % kernel)))
      self % biases = reshape(params(n + 1 :), shape(self % biases))
    end associate

  end subroutine set_params

end submodule nf_locally_connected1d_layer_submodule
