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

    real, allocatable :: output(:,:,:)

  contains

    procedure :: init
    procedure :: forward
    procedure :: backward

  end type maxpool2d_layer

  interface maxpool2d_layer
    pure module function maxpool2d_layer_cons(pool_size, stride) result(res)
      integer, intent(in) :: pool_size
      integer, intent(in) :: stride
      type(maxpool2d_layer) :: res
    end function maxpool2d_layer_cons
  end interface maxpool2d_layer

  interface

    module subroutine init(self, input_shape)
      class(maxpool2d_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init

    pure module subroutine forward(self, input)
      class(maxpool2d_layer), intent(in out) :: self
      real, intent(in) :: input(:,:,:)
    end subroutine forward

    module subroutine backward(self, input, gradient)
      class(maxpool2d_layer), intent(in out) :: self
      real, intent(in) :: input(:,:,:)
      real, intent(in) :: gradient(:,:,:)
    end subroutine backward

  end interface

end module nf_maxpool2d_layer
