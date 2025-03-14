module nf_maxpool3d_layer

    !! This module provides the 3D maxpooling layer.
  
    use nf_base_layer, only: base_layer
    implicit none
  
    private
    public :: maxpool3d_layer
  
    type, extends(base_layer) :: maxpool3d_layer
  
      integer :: channels
      integer :: depth
      integer :: width
      integer :: height
      integer :: pool_size
      integer :: stride
  
      ! Locations (as input matrix indices) of the maximum values
      ! in the depth (z), width (x), and height (y) dimensions
      integer, allocatable :: maxloc_x(:,:,:,:)
      integer, allocatable :: maxloc_y(:,:,:,:)
      integer, allocatable :: maxloc_z(:,:,:,:)
  
      real, allocatable :: gradient(:,:,:,:)
      real, allocatable :: output(:,:,:,:)
  
    contains
  
      procedure :: init
      procedure :: forward
      procedure :: backward
  
    end type maxpool3d_layer
  
    interface maxpool3d_layer
      pure module function maxpool3d_layer_cons(pool_size, stride) result(res)
        !! `maxpool3d` constructor function
        integer, intent(in) :: pool_size
          !! Depth, width, and height of the pooling window
        integer, intent(in) :: stride
          !! Stride of the pooling window
        type(maxpool3d_layer) :: res
      end function maxpool3d_layer_cons
    end interface maxpool3d_layer
  
    interface
  
      module subroutine init(self, input_shape)
        !! Initialize the `maxpool3d` layer instance with an input shape.
        class(maxpool3d_layer), intent(in out) :: self
          !! `maxpool3d_layer` instance
        integer, intent(in) :: input_shape(:)
          !! Array shape of the input layer
      end subroutine init
  
      pure module subroutine forward(self, input)
        !! Run a forward pass of the `maxpool3d` layer.
        class(maxpool3d_layer), intent(in out) :: self
          !! `maxpool3d_layer` instance
        real, intent(in) :: input(:,:,:,:)
          !! Input data (output of the previous layer)
      end subroutine forward
  
      pure module subroutine backward(self, input, gradient)
        !! Run a backward pass of the `maxpool3d` layer.
        class(maxpool3d_layer), intent(in out) :: self
          !! `maxpool3d_layer` instance
        real, intent(in) :: input(:,:,:,:)
          !! Input data (output of the previous layer)
        real, intent(in) :: gradient(:,:,:,:)
          !! Gradient from the downstream layer
      end subroutine backward
  
    end interface
  
  end module nf_maxpool3d_layer
  