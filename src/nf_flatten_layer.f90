module nf_flatten_layer

  !! This module provides the concrete flatten layer type.
  !! It is used internally by the layer type.
  !! It is not intended to be used directly by the user.

  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: flatten_layer

  type, extends(base_layer) :: flatten_layer

    !! Concrete implementation of a flatten (3-d to 1-d) layer.

    integer :: input_shape
    integer :: output_size

    real, allocatable :: output(:)

  contains

    procedure :: backward
    procedure :: forward
    procedure :: init

  end type flatten_layer

  interface flatten_layer
    elemental module function flatten_layer_cons() result(res)
      !! This function returns the `flatten_layer` instance.
      type(flatten_layer) :: res
        !! `flatten_layer` instance
    end function flatten_layer_cons
  end interface flatten_layer

  interface

    pure module subroutine backward(self, input, gradient)
      !! Apply the backward gradient descent pass.
      !! Only weight and bias gradients are updated in this subroutine,
      !! while the weights and biases themselves are untouched.
      class(flatten_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:,:,:)
        !! Input from the previous layer
      real, intent(in) :: gradient(:)
        !! Gradient from the next layer
    end subroutine backward

    pure module subroutine forward(self, input)
      !! Propagate forward the layer.
      !! Calling this subroutine updates the values of a few data components
      !! of `flatten_layer` that are needed for the backward pass.
      class(flatten_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:,:,:)
        !! Input from the previous layer
    end subroutine forward

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(flatten_layer), intent(in out) :: self
        !! Dense layer instance
      integer, intent(in) :: input_shape(:)
        !! Shape of the input layer
    end subroutine init

  end interface

end module nf_flatten_layer
