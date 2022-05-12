module nf_conv2d_layer

  !! This modules provides a 2-d convolutional `conv2d_layer` type.

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

    real, allocatable :: biases(:) ! size(filters)
    real, allocatable :: kernel(:,:,:,:) ! filters x channels x window x window
    real, allocatable :: output(:,:,:) ! filters x output_width * output_height

  contains

    procedure :: init
    procedure :: forward
    procedure :: backward

  end type conv2d_layer

  interface conv2d_layer
    pure module function conv2d_layer_cons(window_size, filters, activation) result(res)
      !! `conv2d_layer` constructor function
      integer, intent(in) :: window_size
      integer, intent(in) :: filters
      character(*), intent(in) :: activation
      type(conv2d_layer) :: res
    end function conv2d_layer_cons
  end interface conv2d_layer

  interface

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(conv2d_layer), intent(in out) :: self
        !! A `conv2d_layer` instance
      integer, intent(in) :: input_shape(:)
        !! Input layer dimensions
    end subroutine init

    pure module subroutine forward(self, input)
      !! Apply a forward pass on the `conv2d` layer.
      class(conv2d_layer), intent(in out) :: self
        !! A `conv2d_layer` instance
      real, intent(in) :: input(:,:,:)
        !! Input data
    end subroutine forward

    module subroutine backward(self, input, gradient)
      !! Apply a backward pass on the `conv2d` layer.
      class(conv2d_layer), intent(in out) :: self
        !! A `conv2d_layer` instance
      real, intent(in) :: input(:,:,:)
        !! Input data (previous layer)
      real, intent(in) :: gradient(:,:,:)
        !! Gradient (next layer)
    end subroutine backward

  end interface

end module nf_conv2d_layer
