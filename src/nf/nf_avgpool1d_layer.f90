module nf_avgpool1d_layer
    !! This module provides the 1-d average pooling layer.
  
    use nf_base_layer, only: base_layer
    implicit none
  
    private
    public :: avgpool1d_layer
  
    type, extends(base_layer) :: avgpool1d_layer
        integer :: channels
        integer :: width      ! Length of the input along the pooling dimension
        integer :: pool_width ! Width of the pooling window
        integer :: stride
  
        ! Gradient for the input (same shape as the input).
        real, allocatable :: gradient(:,:)
        ! Output after pooling (dimensions: (channels, new_width)).
        real, allocatable :: output(:,:)
    contains
        procedure :: init
        procedure :: forward
        procedure :: backward
    end type avgpool1d_layer
  
    interface avgpool1d_layer
        pure module function avgpool1d_layer_cons(pool_width, stride) result(res)
            !! `avgpool1d` constructor function.
            integer, intent(in) :: pool_width
                !! Width of the pooling window.
            integer, intent(in) :: stride
                !! Stride of the pooling window.
            type(avgpool1d_layer) :: res
        end function avgpool1d_layer_cons
    end interface avgpool1d_layer
  
    interface
        module subroutine init(self, input_shape)
            !! Initialize the `avgpool1d` layer instance with an input shape.
            class(avgpool1d_layer), intent(in out) :: self
                !! `avgpool1d_layer` instance.
            integer, intent(in) :: input_shape(:)
                !! Array shape of the input layer, expected as (channels, width).
        end subroutine init
  
        pure module subroutine forward(self, input)
            !! Run a forward pass of the `avgpool1d` layer.
            class(avgpool1d_layer), intent(in out) :: self
                !! `avgpool1d_layer` instance.
            real, intent(in) :: input(:,:)
                !! Input data (output of the previous layer), with shape (channels, width).
        end subroutine forward
  
        pure module subroutine backward(self, input, gradient)
            !! Run a backward pass of the `avgpool1d` layer.
            class(avgpool1d_layer), intent(in out) :: self
                !! `avgpool1d_layer` instance.
            real, intent(in) :: input(:,:)
                !! Input data (output of the previous layer).
            real, intent(in) :: gradient(:,:)
                !! Gradient from the downstream layer, with shape (channels, pooled width).
        end subroutine backward
    end interface
  
  end module nf_avgpool1d_layer
  