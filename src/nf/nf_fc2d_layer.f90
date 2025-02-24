module nf_fc2d_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer
  use nf_linear2d_layer, only: linear2d_layer

  implicit none

  private
  public :: fc2d_layer

  type, extends(base_layer) :: fc2d_layer
    !! Fully Connected 2D Layer
    !! Two Linear layers with an activation function in between
    integer :: sequence_length, model_dimension, hidden_size, output_size

    type(linear2d_layer) :: in_proj
    type(linear2d_layer) :: out_proj

    class(activation_function), allocatable :: activation

    real, allocatable :: gradient(:, :)
    real, allocatable :: in_proj_input(:, :)
    real, allocatable :: out_proj_input(:, :)

    real, allocatable :: output(:, :)

  contains
    procedure :: backward
    procedure :: forward
    procedure :: get_num_params
    procedure :: get_params
    procedure :: get_gradients
    procedure :: set_params
    procedure :: init
  end type fc2d_layer

  interface fc2d_layer
    module function fc2d_layer_cons(hidden_size, output_size, activation) result(res)
      !! This function returns the `fc2d_layer` instance.
      integer, intent(in) :: hidden_size, output_size
      class(activation_function), intent(in) :: activation
      type(fc2d_layer) :: res
    end function fc2d_layer_cons
  end interface fc2d_layer

  interface
    module subroutine init(self, input_shape)
      class(fc2d_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init

    pure module subroutine forward(self, input)
      class(fc2d_layer), intent(in out) :: self
      real, intent(in) :: input(:, :)
    end subroutine forward

    pure module subroutine backward(self, input, gradient)
      class(fc2d_layer), intent(in out) :: self
      real, intent(in) :: input(:, :)
      real, intent(in) :: gradient(:, :)
    end subroutine backward

    elemental module function get_num_params(self) result(num_params)
      class(fc2d_layer), intent(in) :: self
      integer :: num_params
    end function get_num_params

    module function get_params(self) result(params)
      class(fc2d_layer), intent(in) :: self
      real, allocatable :: params(:)
    end function get_params

    module function get_gradients(self) result(gradients)
      class(fc2d_layer), intent(in), target :: self
      real, allocatable :: gradients(:)
    end function get_gradients

    module subroutine set_params(self, params)
      class(fc2d_layer), intent(in out) :: self
      real, intent(in) :: params(:)
    end subroutine set_params
  end interface
end module nf_fc2d_layer
