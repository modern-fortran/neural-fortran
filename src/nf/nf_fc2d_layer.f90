module nf_fc2d_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer
  use nf_linear2d_layer, only: linear2d_layer

  implicit none

  private
  public :: fc2d_layer

  type, extends(base_layer) :: fc2d_layer
    integer :: sequence_length, hidden_size, model_dimension

    type(linear2d_layer) :: in_proj
    type(linear2d_layer) :: out_proj

    class(activation_function), allocatable :: activation

    real, allocatable :: gradient(:, :)
    real, allocatable :: in_proj_input(:, :)
    real, allocatable :: out_proj_input(:, :)

    real, allocatable :: output(:, :)

  contains
!    procedure :: backward
    procedure :: forward
!    procedure :: get_num_params
!    procedure :: get_params
!    procedure :: get_gradients
!    procedure :: set_params
    procedure :: init
  end type fc2d_layer

  interface fc2d_layer
    module function fc2d_layer_cons(hidden_size, activation) result(res)
      !! This function returns the `fc2d_layer` instance.
      integer, intent(in) :: hidden_size
      class(activation_function), intent(in) :: activation
      type(fc2d_layer) :: res
    end function fc2d_layer_cons
  end interface fc2d_layer

contains
  module function fc2d_layer_cons(hidden_size, activation) result(res)
    !! This function returns the `fc2d_layer` instance.
    integer, intent(in) :: hidden_size
    class(activation_function), intent(in) :: activation
    type(fc2d_layer) :: res

    res % hidden_size = hidden_size
    res % activation_name = activation % get_name()
    allocate(res % activation, source = activation)
  end function fc2d_layer_cons

  module subroutine init(self, input_shape)
    class(fc2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    if (size(input_shape) /= 2) then
      error stop "fc2d_layer accepts 2D input"
    end if

    self % sequence_length = input_shape(1)
    self % model_dimension = input_shape(2)

    self % in_proj = linear2d_layer(self % hidden_size)
    call self % in_proj % init([self % sequence_length, self % model_dimension])

    self % out_proj = linear2d_layer(self % model_dimension)
    call self % out_proj % init([self % sequence_length, self % hidden_size])

    allocate(self % in_proj_input(self % sequence_length, self % model_dimension))
    allocate(self % out_proj_input(self % sequence_length, self % hidden_size))

    allocate(self % output(self % sequence_length, self % model_dimension))
  end subroutine init

  pure module subroutine forward(self, input)
    class(fc2d_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    integer :: i

    self % in_proj_input = input
    call self % in_proj % forward(input)

    do concurrent(i = 1: self % sequence_length)
      self % out_proj_input(i, :) = self % activation % eval_1d(self % in_proj % output(i, :))
    end do

    call self % out_proj % forward(self % out_proj_input)
    self % output = self % out_proj % output
  end subroutine forward
end module nf_fc2d_layer
