module nf_conv2d_layer

  !! This is a placeholder module that will later define a concrete conv2d
  !! layer type.

  use nf_base_layer, only: base_layer
  implicit none

  private
  public :: conv2d_layer

  type, extends(base_layer) :: conv2d_layer

    integer :: width
    integer :: height
    integer :: channels
    integer :: window_size
    integer :: filters

    real, allocatable :: biases(:) ! as many as there are filters
    real, allocatable :: kernel(:,:,:,:)
    real, allocatable :: output(:,:,:)

  contains

    procedure :: init
    procedure :: forward
    procedure :: backward

  end type conv2d_layer

  interface conv2d_layer
    pure module function conv2d_layer_cons(window_size, filters, activation) result(res)
      integer, intent(in) :: window_size
      integer, intent(in) :: filters
      character(*), intent(in) :: activation
      type(conv2d_layer) :: res
    end function conv2d_layer_cons
  end interface conv2d_layer

  interface

    module subroutine init(self, input_shape)
      class(conv2d_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init

    pure module subroutine forward(self, input)
      class(conv2d_layer), intent(in out) :: self
      real, intent(in) :: input(:,:,:)
    end subroutine forward

    module subroutine backward(self, input, gradient)
      class(conv2d_layer), intent(in out) :: self
      real, intent(in) :: input(:,:,:)
      real, intent(in) :: gradient(:,:,:)
    end subroutine backward

  end interface

end module nf_conv2d_layer
