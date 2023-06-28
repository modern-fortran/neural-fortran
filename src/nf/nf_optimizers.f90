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
  public :: optimizer_base_type, sgd, rms

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
    real :: momentum = 0
    logical :: nesterov = .false.
  contains
    procedure :: minimize => minimize_sgd
  end type sgd

  type, extends(optimizer_base_type) :: rms
    !! RMSProp optimizer
    real :: decay_rate = 0.9
    real :: epsilon = 1e-8
  contains
    procedure :: minimize => rmsprop_optimizer
  end type rms

  contains

  elemental subroutine minimize_sgd(self, param, gradient)
    !! Concrete implementation of a stochastic gradient descent optimizer
    !! update rule.
    class(sgd), intent(in) :: self
    real, intent(inout) :: param
    real, intent(in) :: gradient
    real, allocatable :: velocity

    if (.not. allocated(velocity)) then
      ! Set initial velocity to zero
      allocate(velocity, mold=param)
      velocity = 0.0
    end if

    if (self % momentum > 0) then
      ! Apply momentum update
      velocity = self % momentum * velocity - self % learning_rate * gradient
      param = param + velocity
    else
      ! Apply regular update
      param = param - self % learning_rate * gradient
    end if

    if (self % nesterov) then
      ! Apply Nesterov update
      velocity = self % momentum * velocity - self % learning_rate * gradient
      param = param + self % momentum * velocity - self % learning_rate * gradient
    end if

  end subroutine minimize_sgd

  elemental subroutine rmsprop_optimizer(self, param, gradient)
    !! Concrete implementation of a RMSProp optimizer
    !! update rule.
    class(rms), intent(in) :: self
    real, intent(inout) :: param
    real, intent(in) :: gradient
    real, allocatable :: rms_gradient

    if (.not. allocated(rms_gradient)) then
      ! Set initial gradients to zero
      allocate(rms_gradient, mold=gradient)
      rms_gradient = 0.0
    end if

    ! Update weights and gradients by RMSProp rule
    rms_gradient = self % decay_rate * rms_gradient + (1 - self % decay_rate) * gradient**2
    param = param - self % learning_rate / sqrt(rms_gradient + self % epsilon) * gradient

  end subroutine rmsprop_optimizer

end module nf_optimizers
