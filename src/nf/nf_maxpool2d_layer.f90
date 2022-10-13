module nf_maxpool2d_layer

  !! This module provides the 2-d maxpooling layer.

  use nf_base_layer, only: base_layer
  implicit none

  private
  public :: maxpool2d_layer

  type, extends(base_layer) :: maxpool2d_layer

    integer :: channels
    integer :: width
    integer :: height
    integer :: pool_size
    integer :: stride

    ! Locations (as input matrix indices) of the maximum values
    ! in the width (x) and height (y) dimensions
    integer, allocatable :: maxloc_x(:,:,:)
    integer, allocatable :: maxloc_y(:,:,:)

    real, allocatable :: gradient(:,:,:)
    real, allocatable :: output(:,:,:)

  contains

    procedure :: init
    procedure :: forward
    procedure :: backward

  end type maxpool2d_layer

  interface maxpool2d_layer
    pure module function maxpool2d_layer_cons(pool_size, stride) result(res)
      !! `maxpool2d` constructor function
      integer, intent(in) :: pool_size
        !! Width and height of the pooling window
      integer, intent(in) :: stride
        !! Stride of the pooling window
      type(maxpool2d_layer) :: res
    end function maxpool2d_layer_cons
  end interface maxpool2d_layer

  interface

    module subroutine init(self, input_shape)
      !! Initialize the `maxpool2d` layer instance with an input shape.
      class(maxpool2d_layer), intent(in out) :: self
        !! `maxpool2d_layer` instance
      integer, intent(in) :: input_shape(:)
        !! Array shape of the input layer
    end subroutine init

    pure module subroutine forward(self, input)
      !! Run a forward pass of the `maxpool2d` layer.
      class(maxpool2d_layer), intent(in out) :: self
        !! `maxpool2d_layer` instance
      real, intent(in) :: input(:,:,:)
        !! Input data (output of the previous layer)
    end subroutine forward

    pure module subroutine backward(self, input, gradient)
      !! Run a backward pass of the `maxpool2d` layer.
      class(maxpool2d_layer), intent(in out) :: self
        !! `maxpool2d_layer` instance
      real, intent(in) :: input(:,:,:)
        !! Input data (output of the previous layer)
      real, intent(in) :: gradient(:,:,:)
        !! Gradient from the downstream layer
    end subroutine backward

  end interface

end module nf_maxpool2d_layer
