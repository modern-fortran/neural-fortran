module nf_batchnorm_layer

  !! This module provides a batch normalization `batchnorm_layer` type.

  use nf_base_layer, only: base_layer
  implicit none

  private
  public :: batchnorm_layer

  type, extends(base_layer) :: batchnorm_layer

    integer :: num_features
    real, allocatable :: gamma(:)
    real, allocatable :: beta(:)
    real, allocatable :: running_mean(:)
    real, allocatable :: running_var(:)
    real, allocatable :: input(:,:)
    real, allocatable :: output(:,:)
    real, allocatable :: gamma_grad(:)
    real, allocatable :: beta_grad(:)
    real, allocatable :: input_grad(:,:)

  contains

    procedure :: forward
    procedure :: backward
    procedure :: get_gradients
    procedure :: get_num_params
    procedure :: get_params
    procedure :: init
    procedure :: set_params

  end type batchnorm_layer

  interface batchnorm_layer
    pure module function batchnorm_layer_cons(num_features) result(res)
      !! `batchnorm_layer` constructor function
      integer, intent(in) :: num_features
      type(batchnorm_layer) :: res
    end function batchnorm_layer_cons
  end interface batchnorm_layer

  interface

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(batchnorm_layer), intent(in out) :: self
        !! A `batchnorm_layer` instance
      integer, intent(in) :: input_shape(:)
        !! Input layer dimensions
    end subroutine init

    pure module subroutine forward(self, input)
      !! Apply a forward pass on the `batchnorm_layer`.
      class(batchnorm_layer), intent(in out) :: self
        !! A `batchnorm_layer` instance
      real, intent(in) :: input(:,:)
        !! Input data
    end subroutine forward

    pure module subroutine backward(self, input, gradient)
      !! Apply a backward pass on the `batchnorm_layer`.
      class(batchnorm_layer), intent(in out) :: self
        !! A `batchnorm_layer` instance
      real, intent(in) :: input(:,:)
        !! Input data (previous layer)
      real, intent(in) :: gradient(:,:)
        !! Gradient (next layer)
    end subroutine backward

    pure module function get_num_params(self) result(num_params)
      !! Get the number of parameters in the layer.
      class(batchnorm_layer), intent(in) :: self
        !! A `batchnorm_layer` instance
      integer :: num_params
        !! Number of parameters
    end function get_num_params

    pure module function get_params(self) result(params)
      !! Return the parameters (gamma, beta, running_mean, running_var) of this layer.
      class(batchnorm_layer), intent(in) :: self
        !! A `batchnorm_layer` instance
      real, allocatable :: params(:)
        !! Parameters to get
    end function get_params

    pure module function get_gradients(self) result(gradients)
      !! Return the gradients of this layer.
      class(batchnorm_layer), intent(in) :: self
        !! A `batchnorm_layer` instance
      real, allocatable :: gradients(:)
        !! Gradients to get
    end function get_gradients

    module subroutine set_params(self, params)
      !! Set the parameters of the layer.
      class(batchnorm_layer), intent(in out) :: self
        !! A `batchnorm_layer` instance
      real, intent(in) :: params(:)
        !! Parameters to set
    end subroutine set_params

  end interface

end module nf_batchnorm_layer
