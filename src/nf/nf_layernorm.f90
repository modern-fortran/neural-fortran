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

  interface
    pure module subroutine forward(self, input)
      class(layernorm_layer), intent(in out) :: self
      real, intent(in) :: input(:, :)
    end subroutine forward

    pure module subroutine backward(self, input, gradient)
      class(layernorm_layer), intent(in out) :: self
      real, intent(in) :: input(:, :)
      real, intent(in) :: gradient(:, :)
    end subroutine backward

    module subroutine init(self, input_shape)
      class(layernorm_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init
  end interface
end module nf_layernorm_layer