module nf_avgpool3d_layer
    !! This module provides the 3-d average pooling layer.

    use nf_base_layer, only: base_layer
    implicit none

    private
    public :: avgpool3d_layer

    type, extends(base_layer) :: avgpool3d_layer
        integer :: channels
        integer :: height      ! Height of the input
        integer :: width       ! Width of the input
        integer :: depth       ! Depth of the input
        integer :: pool_width  ! Pooling window size (width)
        integer :: pool_height ! Pooling window size (height)
        integer :: pool_depth  ! Pooling window size (depth)
        integer :: stride    ! Stride (height, width, depth)

        ! Gradient for the input (same shape as the input: channels, width, height, depth).
        real, allocatable :: gradient(:,:,:,:)
        ! Output after pooling (dimensions: (channels, new_width, new_height, new_depth)).
        real, allocatable :: output(:,:,:,:)
    contains
        procedure :: init
        procedure :: forward
        procedure :: backward
    end type avgpool3d_layer

    interface avgpool3d_layer
        pure module function avgpool3d_layer_cons(pool_width, pool_height, pool_depth, stride) result(res)
            !! `avgpool3d` constructor function.
            integer, intent(in) :: pool_width
                !! Pooling window size (width).
            integer, intent(in) :: pool_height
                !! Pooling window size (height).
            integer, intent(in) :: pool_depth
                !! Pooling window size (depth).
            integer, intent(in) :: stride
                !! Stride (height, width, depth).
            type(avgpool3d_layer) :: res
        end function avgpool3d_layer_cons
    end interface avgpool3d_layer

    interface
        module subroutine init(self, input_shape)
            !! Initialize the `avgpool3d` layer instance with an input shape.
            class(avgpool3d_layer), intent(in out) :: self
                !! `avgpool3d_layer` instance.
            integer, intent(in) :: input_shape(:)
                !! Array shape of the input layer, expected as (channels, width, height, depth).
        end subroutine init

        pure module subroutine forward(self, input)
            !! Run a forward pass of the `avgpool3d` layer.
            class(avgpool3d_layer), intent(in out) :: self
                !! `avgpool3d_layer` instance.
            real, intent(in) :: input(:,:,:,:)
                !! Input data (output of the previous layer), with shape (channels, width, height, depth).
        end subroutine forward

        pure module subroutine backward(self, input, gradient)
            !! Run a backward pass of the `avgpool3d` layer.
            class(avgpool3d_layer), intent(in out) :: self
                !! `avgpool3d_layer` instance.
            real, intent(in) :: input(:,:,:,:)
                !! Input data (output of the previous layer).
            real, intent(in) :: gradient(:,:,:,:)
                !! Gradient from the downstream layer, with shape (channels, pooled_width, pooled_height, pooled_depth).
        end subroutine backward
    end interface

end module nf_avgpool3d_layer
