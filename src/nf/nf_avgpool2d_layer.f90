module nf_avgpool2d_layer
    !! This module provides the 2-d average pooling layer.
  
    use nf_base_layer, only: base_layer
    implicit none
  
    private
    public :: avgpool2d_layer
  
    type, extends(base_layer) :: avgpool2d_layer
        integer :: channels
        integer :: height      ! Height of the input
        integer :: width       ! Width of the input
        integer :: pool_width  ! Pooling window size (width)
        integer :: pool_height ! Pooling window size (height)
        integer :: stride    ! Stride (height, width)
  
        ! Gradient for the input (same shape as the input: channels, height, width).
        real, allocatable :: gradient(:,:,:)
        ! Output after pooling (dimensions: (channels, new_height, new_width)).
        real, allocatable :: output(:,:,:)
    contains
        procedure :: init
        procedure :: forward
        procedure :: backward
    end type avgpool2d_layer
  
    interface avgpool2d_layer
        pure module function avgpool2d_layer_cons(pool_width, pool_height, stride) result(res)
            !! `avgpool2d` constructor function.
            integer, intent(in) :: pool_width
                !! Pooling window size (width).
            integer, intent(in) :: pool_height
                !! Pooling window size (height).
            integer, intent(in) :: stride
                !! Stride (height, width).
            type(avgpool2d_layer) :: res
        end function avgpool2d_layer_cons
    end interface avgpool2d_layer
  
    interface
        module subroutine init(self, input_shape)
            !! Initialize the `avgpool2d` layer instance with an input shape.
            class(avgpool2d_layer), intent(in out) :: self
                !! `avgpool2d_layer` instance.
            integer, intent(in) :: input_shape(:)
                !! Array shape of the input layer, expected as (channels, height, width).
        end subroutine init
  
        pure module subroutine forward(self, input)
            !! Run a forward pass of the `avgpool2d` layer.
            class(avgpool2d_layer), intent(in out) :: self
                !! `avgpool2d_layer` instance.
            real, intent(in) :: input(:,:,:)
                !! Input data (output of the previous layer), with shape (channels, height, width).
        end subroutine forward
  
        pure module subroutine backward(self, input, gradient)
            !! Run a backward pass of the `avgpool2d` layer.
            class(avgpool2d_layer), intent(in out) :: self
                !! `avgpool2d_layer` instance.
            real, intent(in) :: input(:,:,:)
                !! Input data (output of the previous layer).
            real, intent(in) :: gradient(:,:,:)
                !! Gradient from the downstream layer, with shape (channels, pooled_height, pooled_width).
        end subroutine backward
    end interface
  
end module nf_avgpool2d_layer
