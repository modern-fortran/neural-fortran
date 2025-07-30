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

    ! temp storages
    real, allocatable, private :: normalized(:, :)
    real, allocatable, private :: one_over_sigma(:, :)
    real, allocatable, private :: gradient_by_gamma_over_sigma(:, :)
  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
    procedure :: get_num_params
    procedure :: get_params
    procedure :: get_params_ptr
    procedure :: get_gradients
    procedure :: get_gradients_ptr
    procedure :: set_params
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

    pure module function get_num_params(self) result(num_params)
      class(layernorm_layer), intent(in) :: self
      integer :: num_params
    end function get_num_params


    module function get_params(self) result(params)
      class(layernorm_layer), intent(in), target :: self
      real, allocatable :: params(:)
    end function get_params


    module subroutine get_params_ptr(self, g_ptr, b_ptr)
      class(layernorm_layer), intent(in), target :: self
      real, pointer, intent(out) :: g_ptr(:), b_ptr(:)
    end subroutine get_params_ptr


    module function get_gradients(self) result(gradients)
      class(layernorm_layer), intent(in), target :: self
      real, allocatable :: gradients(:)
    end function get_gradients


    module subroutine get_gradients_ptr(self, dg_ptr, db_ptr)
      class(layernorm_layer), intent(in), target :: self
      real, pointer, intent(out) :: dg_ptr(:), db_ptr(:)
    end subroutine get_gradients_ptr


    module subroutine set_params(self, params)
      class(layernorm_layer), intent(in out) :: self
      real, intent(in), target :: params(:)
    end subroutine set_params
  end interface
end module nf_layernorm_layer