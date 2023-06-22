module nf_optimizers

  !! This module provides optimizer types to pass to the network train or update
  !! methods. The implementation is based on an abstract optimizer base type
  !! which has a required minimize method. The minimize method is an elemental
  !! subroutine to allow operating in-place on arrays of network parameters
  !! (weights/kernels and biases) of arbitrary ranks. An implementation of a new
  !! optimizer thus requires writing a concrete optimizer type that extends the
  !! abstract optimizer base type, and that implements a concrete minimize
  !! method that accepts a scalar or array of network parameters to update and
  !! the corresponding loss gradients.

  implicit none

  private
  public :: optimizer_base_type, sgd

  type, abstract :: optimizer_base_type
    real :: learning_rate = 1
  contains
    procedure(minimize), deferred :: minimize
  end type optimizer_base_type

  abstract interface
    elemental subroutine minimize(self, param, gradient)
      import :: optimizer_base_type
      class(optimizer_base_type), intent(in) :: self
      real, intent(inout) :: param
      real, intent(in) :: gradient
    end subroutine minimize
  end interface

  type, extends(optimizer_base_type) :: sgd
    !! Stochastic Gradient Descent optimizer
    real :: momentum = 0 !TODO
    logical :: nesterov = .false. !TODO
  contains
    procedure :: minimize => minimize_sgd
  end type sgd

contains

  elemental subroutine minimize_sgd(self, param, gradient)
    !! Concrete implementation of a stochastic gradient descent optimizer
    !! update rule.
    class(sgd), intent(in) :: self
      !! Optimizer instance
    real, intent(inout) :: param
      !! Network parameter (i.e. weight or bias) to update
    real, intent(in) :: gradient
      !! Loss gradient with respect to the parameter (dL/dw or dL/db)
    ! TODO Implement momentum and Nesterov options
    ! TODO (see https://keras.io/api/optimizers/sgd/)
    param = param - self % learning_rate * gradient
  end subroutine minimize_sgd

end module nf_optimizers
