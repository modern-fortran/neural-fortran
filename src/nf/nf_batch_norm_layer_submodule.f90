submodule(nf_batch_norm_layer) nf_batch_norm_layer_submodule

  use nf_base_layer, only: base_layer
  implicit none

contains

  pure module function batch_norm_layer_cons(size) result(res)
    implicit none
    integer, intent(in) :: size
    type(batch_norm_layer) :: res

    res % size = size
    allocate(res % gamma(size), source=1.0)
    allocate(res % beta(size))
    allocate(res % running_mean(size), source=0.0)
    allocate(res % running_var(size), source=1.0)
    allocate(res % input(size, size))
    allocate(res % output(size, size))
    allocate(res % gamma_grad(size))
    allocate(res % beta_grad(size))
    allocate(res % input_grad(size, size))

  end function batch_norm_layer_cons

  module subroutine init(self, input_shape)
    implicit none
    class(batch_norm_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % input = 0
    self % output = 0

    ! Initialize gamma, beta, running_mean, and running_var
    self % gamma = 1.0
    self % beta = 0.0
    self % running_mean = 0.0
    self % running_var = 1.0

  end subroutine init

  pure module subroutine forward(self, input)
    implicit none
    class(batch_norm_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    real, allocatable :: normalized_input(:,:)

    ! Store input for backward pass
    self % input = input

    ! Calculate the normalized input
    normalized_input = (input - reshape(self % running_mean, shape(input, 1))) * &
                      reshape(self % gamma, shape(input, 1)) / &
                      sqrt(reshape(self % running_var, shape(input, 1)) + 1.0e-8)

    ! Batch normalization forward pass
    self % output = normalized_input + reshape(self % beta, shape(input, 1))

    ! Deallocate temporary array
    deallocate(normalized_input)

  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    implicit none
    class(batch_norm_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    real, intent(in) :: gradient(:,:)

    ! Calculate gradients for gamma, beta
    self % gamma_grad = sum(gradient * (input - reshape(self % running_mean, shape(input, 1))) / &
                            sqrt(reshape(self % running_var, shape(input, 1)) + 1.0e-8), dim=2)
    self % beta_grad = sum(gradient, dim=2)

    ! Calculate gradients for input
    self % input_grad = gradient * reshape(self % gamma, shape(gradient)) / &
                      sqrt(reshape(self % running_var, shape(input, 1)) + 1.0e-8)

  end subroutine backward

  pure module function get_num_params(self) result(num_params)
    class(batch_norm_layer), intent(in) :: self
    integer :: num_params
    num_params = 2 * self % size
  end function get_num_params

  pure module function get_params(self) result(params)
    class(batch_norm_layer), intent(in) :: self
    real, allocatable :: params(:)
    params = [self % gamma, self % beta]
  end function get_params

  pure module function get_gradients(self) result(gradients)
    class(batch_norm_layer), intent(in) :: self
    real, allocatable :: gradients(:)
    gradients = [self % gamma_grad, self % beta_grad]
  end function get_gradients

  module subroutine set_params(self, params)
    class(batch_norm_layer), intent(in out) :: self
    real, intent(in) :: params(:)
    self % gamma = params(1:self % size)
    self % beta = params(self % size+1:2*self % size)
  end subroutine set_params

end submodule nf_batch_norm_layer_submodule
