module nf_layernorm_layer
  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: layernorm_layer

  type, extends(base_layer) :: layernorm_layer
    !! Layer Normalization
    !! ((x − mean(x)) / sqrt(variance(x) + eps) * gamma + beta
    !! Based upon `Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton(2016)`:
    !! https://arxiv.org/abs/1607.06450v1
    integer :: sequence_length
    integer :: model_dimension

    real :: eps
    real, allocatable :: gamma(:)
    real, allocatable :: beta(:)

    real, allocatable :: d_gamma(:)
    real, allocatable :: d_beta(:)
    real, allocatable :: gradient(:, :)

    real, allocatable :: mu(:, :)
    real, allocatable :: sigma(:)

    real, allocatable :: output(:, :)

  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
  end type layernorm_layer

  interface layernorm_layer
    module function layernorm_layer_cons() &
      result(res)
      type(layernorm_layer) :: res
    end function layernorm_layer_cons
  end interface layernorm_layer

contains
  module function layernorm_layer_cons() &
    result(res)
    type(layernorm_layer) :: res

    res % eps = 1e-5
  end function layernorm_layer_cons

  pure module subroutine forward(self, input)
    class(layernorm_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    real, allocatable :: normalized(:, :)
    integer :: i

    allocate(normalized(self % sequence_length, self % model_dimension))

    ! mu = x - MEAN_last_dim(x)
    do concurrent(i = 1: self % model_dimension)
      self % mu(:, i) = input(:, i) - (sum(input, dim=2) / self % model_dimension)
    end do

    ! square root of variance shifted be eps
    self % sigma = sqrt((sum(self % mu ** 2, dim=2) / self % model_dimension) + self % eps)

    ! normalize mu by variance by first axis
    do concurrent(i = 1: self % model_dimension)
      normalized(:, i) = self % mu(:, i) / self % sigma
    end do

    ! forward through trainable params gamma and beta
    do concurrent(i = 1: self % sequence_length)
      self % output(i, :) = normalized(i, :) * self % gamma + self % beta
    end do

    deallocate(normalized)
  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    class(layernorm_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: gradient(:, :)
    real, allocatable :: one_over_sigma(:, :)
    real, allocatable :: gradient_by_gamma_over_sigma(:, :)

    allocate(one_over_sigma(self % sequence_length, self % model_dimension))
    allocate(gradient_by_gamma_over_sigma(self % sequence_length, self % model_dimension))

    one_over_sigma = (1 / spread(self % sigma, dim=2, ncopies=self % model_dimension))
    gradient_by_gamma_over_sigma = &
        gradient &
        * spread(self % gamma, dim=1, ncopies=self % sequence_length) &
        * one_over_sigma

    ! d_output/d_gamma = sum(d_output/d_y * mu/sigma)
    self % d_gamma = sum(gradient * self % mu * one_over_sigma, dim=1)

    ! d_output/d_beta = sum(d_output/d_y) * 1
    self % d_beta = sum(gradient, dim=1)

    ! From this article:
    ! https://robotchinwag.com/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/
    ! d_output/d_x = d_output/d_y * gamma/sigma
    !     - d_output/d_y
    !     - sum(d_output/d_y * gamma/sigma) / len
    !     - mu * sum(d_output/d_y * gamma * mu * sigma^(03)) / len
    self % gradient = &
        gradient_by_gamma_over_sigma &
        - spread(&
            sum(gradient_by_gamma_over_sigma, dim=2),&
            dim=2,&
            ncopies=self % model_dimension&
          ) / self % model_dimension &
        - self % mu * spread(&
            sum(gradient_by_gamma_over_sigma * self % mu * (one_over_sigma ** 2), dim=2),&
            dim=2,&
            ncopies=self % model_dimension&
          ) / self % model_dimension

    deallocate(one_over_sigma)
    deallocate(gradient_by_gamma_over_sigma)
  end subroutine backward

  module subroutine init(self, input_shape)
    class(layernorm_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    if (size(input_shape) /= 2) then
      error stop "LayerNorm Layer accepts 2D input"
    end if
    self % sequence_length = input_shape(1)
    self % model_dimension = input_shape(2)

    ! default initialization from PyTorch
    allocate(self % gamma(self % model_dimension))
    self % gamma = 1.
    allocate(self % beta(self % model_dimension))
    self % beta = 0.

    allocate(self % d_gamma(self % model_dimension))
    allocate(self % d_beta(self % model_dimension))
    allocate(self % gradient(self % sequence_length, self % model_dimension))

    allocate(self % mu(self % sequence_length, self % model_dimension))
    allocate(self % sigma(self % sequence_length))

    allocate(self % output(self % sequence_length, self % model_dimension))
  end subroutine init
end module nf_layernorm_layer