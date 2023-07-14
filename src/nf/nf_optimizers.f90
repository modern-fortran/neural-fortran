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
  public :: optimizer_base_type, sgd, rmsprop

  type, abstract :: optimizer_base_type
    real :: learning_rate = 0.01
  contains
    procedure(init), deferred :: init
    procedure(minimize), deferred :: minimize
  end type optimizer_base_type

  abstract interface

    impure elemental subroutine init(self, num_params)
      import :: optimizer_base_type
      class(optimizer_base_type), intent(inout) :: self
      integer, intent(in) :: num_params
    end subroutine init

    pure subroutine minimize(self, param, gradient)
      import :: optimizer_base_type
      class(optimizer_base_type), intent(inout) :: self
      real, intent(inout) :: param(:)
      real, intent(in) :: gradient(:)
    end subroutine minimize

  end interface

  type, extends(optimizer_base_type) :: sgd
    !! Stochastic Gradient Descent optimizer
    real :: momentum = 0
    logical :: nesterov = .false.
    real, allocatable :: velocity(:)
  contains
    procedure :: init => init_sgd
    procedure :: minimize => minimize_sgd
  end type sgd

  type, extends(optimizer_base_type) :: rmsprop
    !! RMSProp optimizer
    real :: decay_rate = 0.9
    real :: epsilon = 1e-8
    real, allocatable :: rms_gradient(:)
  contains
    procedure :: init => init_rmsprop
    procedure :: minimize => minimize_rmsprop
  end type rmsprop

contains

  impure elemental subroutine init_sgd(self, num_params)
    class(sgd), intent(inout) :: self
    integer, intent(in) :: num_params
    if (self % momentum > 0 .and. .not. allocated(self % velocity)) then
      allocate(self % velocity(num_params))
      self % velocity = 0
    end if
  end subroutine init_sgd


  pure subroutine minimize_sgd(self, param, gradient)
    !! Concrete implementation of a stochastic gradient descent optimizer
    !! update rule.
    class(sgd), intent(inout) :: self
    real, intent(inout) :: param(:)
    real, intent(in) :: gradient(:)

    if (self % momentum > 0) then
      ! Apply momentum update
      self % velocity = self % momentum * self % velocity &
        - self % learning_rate * gradient
      if (self % nesterov) then
        ! Apply Nesterov update
        param = param + self % momentum * self % velocity &
          - self % learning_rate * gradient
      else
        param = param + self % velocity
      end if
    else
      ! Apply regular update
      param = param - self % learning_rate * gradient
    end if

  end subroutine minimize_sgd


  impure elemental subroutine init_rmsprop(self, num_params)
    class(rmsprop), intent(inout) :: self
    integer, intent(in) :: num_params
    if (.not. allocated(self % rms_gradient)) then
      allocate(self % rms_gradient(num_params))
      self % rms_gradient = 0
    end if
  end subroutine init_rmsprop


  pure subroutine minimize_rmsprop(self, param, gradient)
    !! Concrete implementation of a RMSProp optimizer update rule.
    class(rmsprop), intent(inout) :: self
    real, intent(inout) :: param(:)
    real, intent(in) :: gradient(:)

    ! Compute the RMS of the gradient using the RMSProp rule
    self % rms_gradient = self % decay_rate * self % rms_gradient &
      + (1 - self % decay_rate) * gradient**2

    ! Update the network parameters based on the new RMS of the gradient
    param = param - self % learning_rate &
      / sqrt(self % rms_gradient + self % epsilon) * gradient

  end subroutine minimize_rmsprop

end module nf_optimizers
