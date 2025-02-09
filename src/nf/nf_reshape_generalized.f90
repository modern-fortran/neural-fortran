module nf_reshape_layer_generalized

    !! This module provides the concrete reshape layer type.
    !! It is used internally by the layer type.
    !! It is not intended to be used directly by the user.
  
    use nf_base_layer, only: base_layer
  
    implicit none
  
    private
    public :: reshape_generalized_layer
  
    type, extends(base_layer) :: reshape_generalized_layer
  
      !! Concrete implementation of a reshape layer type
      !! It implements reshaping for arbitrary ranks.
  
      integer, allocatable :: input_shape(:)
      integer, allocatable :: output_shape(:)
      real, allocatable :: gradient(:)
      real, allocatable :: output(:)
  
    contains
  
      procedure :: backward
      procedure :: forward
      procedure :: init
  
    end type reshape_generalized_layer
  
    interface reshape_generalized_layer
      pure module function reshape_layer_cons(output_shape) result(res)
        !! This function returns the `reshape_layer` instance.
        integer, intent(in) :: output_shape(:)
          !! The shape of the output
        type(reshape_generalized_layer) :: res
          !! reshape_layer instance
      end function reshape_layer_cons
    end interface reshape_generalized_layer
  
    interface
  
      pure module subroutine backward(self, input, gradient)
        !! Apply the backward pass for the reshape layer.
        !! This is just flattening to a rank-1 array.
        class(reshape_generalized_layer), intent(in out) :: self
          !! Dense layer instance
        real, intent(in) :: input(:)
          !! Input from the previous layer
        real, intent(in) :: gradient(..)
          !! Gradient from the next layer
      end subroutine backward
  
      pure module subroutine forward(self, input)
        !! Apply the forward pass for the reshape layer.
        !! This is reshaping from input rank to output rank.
        class(reshape_generalized_layer), intent(in out) :: self
          !! Dense layer instance
        real, intent(in) :: input(:)
          !! Input from the previous layer
      end subroutine forward
  
      module subroutine init(self, input_shape)
        !! Initialize the layer data structures.
        !!
        !! This is a deferred procedure from the `base_layer` abstract type.
        class(reshape_generalized_layer), intent(in out) :: self
          !! Dense layer instance
        integer, intent(in) :: input_shape(:)
          !! Shape of the input layer
      end subroutine init
  
    end interface
  
  end module nf_reshape_layer_generalized
  