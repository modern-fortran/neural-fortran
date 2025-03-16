module nf_avgpool3d_layer
    !! This module provides the 3-d average pooling layer.
  
    use nf_base_layer, only: base_layer
    implicit none
  
    private
    public :: avgpool3d_layer
  
    type, extends(base_layer) :: avgpool3d_layer
        integer :: channels
        integer :: depth       ! Depth of the input
        integer :: height      ! Height of the input
        integer :: width       ! Width of the input
        integer :: pool_size(3)  ! Pooling window size (depth, height, width)
        integer :: stride(3)     ! Stride (depth, height, width)
  
        ! Gradient for the input (same shape as the input).
        real, allocatable :: gradient(:,:,:,:)
        ! Output after pooling (dimensions: (channels, new_depth, new_height, new_width)).
        real, allocatable :: output(:,:,:,:)
    contains
        procedure :: init
        procedure :: forward
        procedure :: backward
    end type avgpool3d_layer
  
    interface avgpool3d_layer
        pure module function avgpool3d_layer_cons(pool_size, stride) result(res)
            !! `avgpool3d` constructor function.
            integer, intent(in) :: pool_size(3)
                !! Depth, height, and width of the pooling window.
            integer, intent(in) :: stride(3)
                !! Stride of the pooling window (depth, height, width).
            type(avgpool3d_layer) :: res
        end function avgpool3d_layer_cons
    end interface avgpool3d_layer
  
    interface
        module subroutine init(self, input_shape)
            !! Initialize the `avgpool3d` layer instance with an input shape.
            class(avgpool3d_layer), intent(in out) :: self
                !! `avgpool3d_layer` instance.
            integer, intent(in) :: input_shape(:)
                !! Array shape of the input layer, expected as (channels, depth, height, width).
        end subroutine init
  
        pure module subroutine forward(self, input)
            !! Run a forward pass of the `avgpool3d` layer.
            class(avgpool3d_layer), intent(in out) :: self
                !! `avgpool3d_layer` instance.
            real, intent(in) :: input(:,:,:,:)
                !! Input data (output of the previous layer), with shape (channels, depth, height, width).
        end subroutine forward
  
        pure module subroutine backward(self, input, gradient)
            !! Run a backward pass of the `avgpool3d` layer.
            class(avgpool3d_layer), intent(in out) :: self
                !! `avgpool3d_layer` instance.
            real, intent(in) :: input(:,:,:,:)
                !! Input data (output of the previous layer).
            real, intent(in) :: gradient(:,:,:,:)
                !! Gradient from the downstream layer, with shape (channels, pooled_depth, pooled_height, pooled_width).
        end subroutine backward
    end interface
  
end module nf_avgpool3d_layer
