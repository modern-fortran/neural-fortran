module nf_dropout_layer

  !! This module provides the concrete dropout layer type.
  !! It is used internally by the layer type.
  !! It is not intended to be used directly by the user.

  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: dropout_layer

  type, extends(base_layer) :: dropout_layer
    !! Concrete implementation of a dropout layer type

    integer :: input_size = 0

    real, allocatable :: output(:)
    real, allocatable :: gradient(:)
    real, allocatable :: mask(:) ! binary mask for dropout

    real :: dropout_rate ! probability of dropping a neuron
    real :: scale ! scale factor to preserve the input sum
    logical :: training = .true.

  contains

    procedure :: backward
    procedure :: forward
    procedure :: init

  end type dropout_layer

  interface dropout_layer
    module function dropout_layer_cons(rate) &
      result(res)
      !! This function returns the `dropout_layer` instance.
      real, intent(in) :: rate
        !! Dropout rate
      type(dropout_layer) :: res
        !! dropout_layer instance
    end function dropout_layer_cons
  end interface dropout_layer

  interface

    pure module subroutine backward(self, input, gradient)
      !! Apply the backward gradient descent pass.
      !! Only weight and bias gradients are updated in this subroutine,
      !! while the weights and biases themselves are untouched.
      class(dropout_layer), intent(in out) :: self
        !! Dropout layer instance
      real, intent(in) :: input(:)
        !! Input from the previous layer
      real, intent(in) :: gradient(:)
        !! Gradient from the next layer
    end subroutine backward

    module subroutine forward(self, input)
      !! Propagate forward the layer.
      !! Calling this subroutine updates the values of a few data components
      !! of `dropout_layer` that are needed for the backward pass.
      class(dropout_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:)
        !! Input from the previous layer
    end subroutine forward

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(dropout_layer), intent(in out) :: self
        !! Dropout layer instance
      integer, intent(in) :: input_shape(:)
        !! Shape of the input layer
    end subroutine init

  end interface

end module nf_dropout_layer
