module nf_linear2d_layer

  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: linear2d_layer

  type, extends(base_layer) :: linear2d_layer
    integer :: sequence_length, in_features, out_features, batch_size

    real, allocatable :: weights(:,:)
    real, allocatable :: biases(:)
    real, allocatable :: output(:,:)
    real, allocatable :: gradient(:,:) ! input gradient
    real, allocatable :: dw(:,:) ! weight gradients
    real, allocatable :: db(:) ! bias gradients

  contains

    procedure :: backward
    procedure :: forward
    procedure :: init
    procedure :: get_num_params
    procedure :: get_params_ptr
    procedure :: get_gradients
    procedure :: get_gradients_ptr

  end type linear2d_layer

  interface linear2d_layer
    module function linear2d_layer_cons(out_features) result(res)
      integer, intent(in) :: out_features
      type(linear2d_layer) :: res
    end function linear2d_layer_cons
  end interface linear2d_layer

  interface
    pure module subroutine forward(self, input)
      class(linear2d_layer), intent(in out) :: self
      real, intent(in) :: input(:,:)
    end subroutine forward

    pure module subroutine backward(self, input, gradient)
      class(linear2d_layer), intent(in out) :: self
      real, intent(in) :: input(:,:)
      real, intent(in) :: gradient(:,:)
    end subroutine backward

    module subroutine init(self, input_shape)
      class(linear2d_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init

    pure module function get_num_params(self) result(num_params)
       class(linear2d_layer), intent(in) :: self
       integer :: num_params
    end function get_num_params

    module subroutine get_params_ptr(self, w_ptr, b_ptr)
      class(linear2d_layer), intent(in), target :: self
      real, pointer, intent(out) :: w_ptr(:), b_ptr(:)
    end subroutine get_params_ptr

    module function get_gradients(self) result(gradients)
      class(linear2d_layer), intent(in), target :: self
      real, allocatable :: gradients(:)
    end function get_gradients

    module subroutine get_gradients_ptr(self, dw_ptr, db_ptr)
      class(linear2d_layer), intent(in), target :: self
      real, pointer, intent(out) :: dw_ptr(:), db_ptr(:)
    end subroutine get_gradients_ptr

  end interface
end module nf_linear2d_layer
