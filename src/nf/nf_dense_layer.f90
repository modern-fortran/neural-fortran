module nf_dense_layer

  !! This module provides the concrete dense layer type.
  !! It is used internally by the layer type.
  !! It is not intended to be used directly by the user.

  use nf_optimizers, only: optimizer_base_type
  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: dense_layer

  type, extends(base_layer) :: dense_layer

    !! Concrete implementation of a dense (fully-connected) layer type

    integer :: input_size
    integer :: output_size

    real, allocatable :: weights(:,:)
    real, allocatable :: biases(:)
    real, allocatable :: z(:) ! matmul(x, w) + b
    real, allocatable :: output(:) ! activation(z)
    real, allocatable :: gradient(:) ! matmul(w, db)
    real, allocatable :: dw(:,:) ! weight gradients
    real, allocatable :: db(:) ! bias gradients

    class(activation_function), allocatable :: activation
    class(optimizer_base_type), allocatable :: optimizer_1d
    class(optimizer_base_type), allocatable :: optimizer_2d

  contains

    procedure :: backward
    procedure :: forward
    procedure :: get_gradients
    procedure :: get_num_params
    procedure :: get_params
    procedure :: init
    procedure :: set_params
    procedure :: apply_optimizer
    procedure :: set_optimizer

  end type dense_layer

  interface dense_layer
    elemental module function dense_layer_cons(output_size, activation) &
      result(res)
      !! This function returns the `dense_layer` instance.
      integer, intent(in) :: output_size
        !! Number of neurons in this layer
      class(activation_function), intent(in) :: activation
        !! Instance of the activation_function to use;
        !! See nf_activation.f90 for available functions.
      type(dense_layer) :: res
        !! dense_layer instance
    end function dense_layer_cons
  end interface dense_layer

  interface

    pure module subroutine backward(self, input, gradient)
      !! Apply the backward gradient descent pass.
      !! Only weight and bias gradients are updated in this subroutine,
      !! while the weights and biases themselves are untouched.
      class(dense_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:)
        !! Input from the previous layer
      real, intent(in) :: gradient(:)
        !! Gradient from the next layer
    end subroutine backward

    pure module subroutine forward(self, input)
      !! Propagate forward the layer.
      !! Calling this subroutine updates the values of a few data components
      !! of `dense_layer` that are needed for the backward pass.
      class(dense_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:)
        !! Input from the previous layer
    end subroutine forward

    pure module function get_num_params(self) result(num_params)
       !! Return the number of parameters in this layer.
       class(dense_layer), intent(in) :: self
         !! Dense layer instance
       integer :: num_params
         !! Number of parameters in this layer
    end function get_num_params

    module function get_params(self) result(params)
      !! Return the parameters (weights and biases) of this layer.
      !! The parameters are ordered as weights first, biases second.
      class(dense_layer), intent(in), target :: self
        !! Dense layer instance
      real, allocatable :: params(:)
        !! Parameters of this layer
    end function get_params

    module function get_gradients(self) result(gradients)
      !! Return the gradients of this layer.
      !! The gradients are ordered as weights first, biases second.
      class(dense_layer), intent(in), target :: self
        !! Dense layer instance
      real, allocatable :: gradients(:)
        !! Gradients of this layer
    end function get_gradients

    module subroutine set_params(self, params)
      !! Set the parameters of this layer.
      !! The parameters are ordered as weights first, biases second.
      class(dense_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in), target :: params(:)
        !! Parameters of this layer
    end subroutine set_params

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(dense_layer), intent(in out) :: self
        !! Dense layer instance
      integer, intent(in) :: input_shape(:)
        !! Shape of the input layer

    end subroutine init

    module subroutine apply_optimizer(self, batch_size)
      class(dense_layer), intent(in out), target :: self
      integer, intent(in) :: batch_size
    end subroutine apply_optimizer

    module subroutine set_optimizer(self, optimizer)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(dense_layer), intent(in out) :: self
        !! Dense layer instance
      class(optimizer_base_type), intent(in), optional :: optimizer

    end subroutine set_optimizer


  end interface

end module nf_dense_layer
