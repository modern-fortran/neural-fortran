module nf_maxpool1d_layer
  !! This module provides the 1-d maxpooling layer.

  use nf_base_layer, only: base_layer
  implicit none

  private
  public :: maxpool1d_layer

  type, extends(base_layer) :: maxpool1d_layer
      integer :: channels
      integer :: width      ! Length of the input along the pooling dimension
      integer :: pool_size
      integer :: stride

      ! Location (as input matrix indices) of the maximum value within each pooling region.
      ! Dimensions: (channels, new_width)
      integer, allocatable :: maxloc(:,:)

      ! Gradient for the input (same shape as the input).
      real, allocatable :: gradient(:,:)
      ! Output after pooling (dimensions: (channels, new_width)).
      real, allocatable :: output(:,:)
  contains
      procedure :: init
      procedure :: forward
      procedure :: backward
  end type maxpool1d_layer

  interface maxpool1d_layer
      pure module function maxpool1d_layer_cons(pool_size, stride) result(res)
          !! `maxpool1d` constructor function.
          integer, intent(in) :: pool_size
              !! Width of the pooling window.
          integer, intent(in) :: stride
              !! Stride of the pooling window.
          type(maxpool1d_layer) :: res
      end function maxpool1d_layer_cons
  end interface maxpool1d_layer

  interface
      module subroutine init(self, input_shape)
          !! Initialize the `maxpool1d` layer instance with an input shape.
          class(maxpool1d_layer), intent(in out) :: self
              !! `maxpool1d_layer` instance.
          integer, intent(in) :: input_shape(:)
              !! Array shape of the input layer, expected as (channels, width).
      end subroutine init

      pure module subroutine forward(self, input)
          !! Run a forward pass of the `maxpool1d` layer.
          class(maxpool1d_layer), intent(in out) :: self
              !! `maxpool1d_layer` instance.
          real, intent(in) :: input(:,:)
              !! Input data (output of the previous layer), with shape (channels, width).
      end subroutine forward

      pure module subroutine backward(self, input, gradient)
          !! Run a backward pass of the `maxpool1d` layer.
          class(maxpool1d_layer), intent(in out) :: self
              !! `maxpool1d_layer` instance.
          real, intent(in) :: input(:,:)
              !! Input data (output of the previous layer).
          real, intent(in) :: gradient(:,:)
              !! Gradient from the downstream layer, with shape (channels, pooled width).
      end subroutine backward
  end interface

end module nf_maxpool1d_layer