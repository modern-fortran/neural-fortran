module nf_maxpool2d_layer

  !! This is a placeholder module that will later define a concrete maxpool2d
  !! layer type.

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

    real, allocatable :: output(:,:,:)

  contains

    procedure :: init
    procedure :: forward
    procedure :: backward

  end type maxpool2d_layer

  interface maxpool2d_layer
    module procedure :: maxpool2d_layer_cons
  end interface maxpool2d_layer

contains

  !TODO move implementations to a submodule

  pure function maxpool2d_layer_cons(pool_size, stride) result(res)
    integer, intent(in) :: pool_size
    integer, intent(in) :: stride
    type(maxpool2d_layer) :: res
    res % pool_size = pool_size
    res % stride = stride
  end function maxpool2d_layer_cons


  subroutine init(self, input_shape)
    class(maxpool2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
    !TODO
  end subroutine init


  subroutine forward(self, input)
    class(maxpool2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    print *, 'Warning: maxpool2d forward pass not implemented'
  end subroutine forward


  subroutine backward(self, input, gradient)
    class(maxpool2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:,:,:)
    print *, 'Warning: maxpool2d backward pass not implemented'
  end subroutine backward

end module nf_maxpool2d_layer
