module nf_reshape_layer

  !! This module provides the concrete reshape layer type.
  !! It is used internally by the layer type.
  !! It is not intended to be used directly by the user.

  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: reshape_layer

  type, extends(base_layer) :: reshape_layer

    !! Concrete implementation of a reshape layer type
    !! It implements only rank-1 to rank-3 reshaping.

    integer :: input_size

    real, allocatable :: output(:,:,:)

  contains

    procedure :: backward
    procedure :: forward
    procedure :: init

  end type reshape_layer

  interface reshape_layer
    elemental module function reshape_layer_cons(output_shape) result(res)
      !! This function returns the `reshape_layer` instance.
      integer, intent(in) :: output_shape(:)
        !! The shape of the output
      type(reshape_layer) :: res
        !! reshape_layer instance
    end function reshape_layer_cons
  end interface reshape_layer

  interface

    pure module subroutine backward(self, input, gradient)
      !! Apply the backward gradient descent pass.
      !! Only weight and bias gradients are updated in this subroutine,
      !! while the weights and biases themselves are untouched.
      class(reshape_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:)
        !! Input from the previous layer
      real, intent(in) :: gradient(:,:,:)
        !! Gradient from the next layer
    end subroutine backward

    pure module subroutine forward(self, input)
      !! Propagate forward the layer.
      !! Calling this subroutine updates the values of a few data components
      !! of `reshape_layer` that are needed for the backward pass.
      class(reshape_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:)
        !! Input from the previous layer
    end subroutine forward

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(reshape_layer), intent(in out) :: self
        !! Dense layer instance
      integer, intent(in) :: input_shape(:)
        !! Shape of the input layer
    end subroutine init

  end interface

end module nf_reshape_layer
