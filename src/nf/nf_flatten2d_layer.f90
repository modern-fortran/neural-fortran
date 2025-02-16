module nf_flatten2d_layer

  !! This module provides the concrete flatten2d layer type.
  !! It is used internally by the layer type.
  !! It is not intended to be used directly by the user.

  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: flatten2d_layer

  type, extends(base_layer) :: flatten2d_layer

    !! Concrete implementation of a flatten2d (2-d to 1-d) layer.

    integer, allocatable :: input_shape(:)
    integer :: output_size

    real, allocatable :: gradient(:,:)
    real, allocatable :: output(:)

  contains

    procedure :: backward
    procedure :: forward
    procedure :: init

  end type flatten2d_layer

  interface flatten2d_layer
    elemental module function flatten2d_layer_cons() result(res)
      !! This function returns the `flatten2d_layer` instance.
      type(flatten2d_layer) :: res
        !! `flatten2d_layer` instance
    end function flatten2d_layer_cons
  end interface flatten2d_layer

  interface

    pure module subroutine backward(self, input, gradient)
      !! Apply the backward pass to the flatten2d layer.
      !! This is a reshape operation from 1-d gradient to 2-d input.
      class(flatten2d_layer), intent(in out) :: self
        !! flatten2d layer instance
      real, intent(in) :: input(:,:)
        !! Input from the previous layer
      real, intent(in) :: gradient(:)
        !! Gradient from the next layer
    end subroutine backward

    pure module subroutine forward(self, input)
      !! Propagate forward the layer.
      !! Calling this subroutine updates the values of a few data components
      !! of `flatten2d_layer` that are needed for the backward pass.
      class(flatten2d_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:,:)
        !! Input from the previous layer
    end subroutine forward

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(flatten2d_layer), intent(in out) :: self
        !! Dense layer instance
      integer, intent(in) :: input_shape(:)
        !! Shape of the input layer
    end subroutine init

  end interface

end module nf_flatten2d_layer
