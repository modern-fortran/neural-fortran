module nf_optimizers

  !! This module provides optimizer types to pass to the network constructor.

  implicit none

  private
  public :: optimizer_base_type, sgd

  type, abstract :: optimizer_base_type
    character(:), allocatable :: name
  contains
    procedure(minimize_interface), deferred :: minimize
  end type optimizer_base_type

  abstract interface
    elemental subroutine minimize_interface(self, weight, gradient)
      import :: optimizer_base_type
      class(optimizer_base_type), intent(in) :: self
      real, intent(inout) :: weight
      real, intent(in) :: gradient
    end subroutine minimize_interface
  end interface

  type, extends(optimizer_base_type) :: sgd
    !! Stochastic Gradient Descent optimizer
    real :: learning_rate
    real :: momentum = 0 !TODO
    logical :: nesterov = .false. !TODO
  contains
    procedure :: minimize => minimize_sgd
  end type sgd

contains

  elemental subroutine minimize_sgd(self, weight, gradient)
    class(sgd), intent(in) :: self
    real, intent(inout) :: weight
    real, intent(in) :: gradient
    weight = weight - self % learning_rate * gradient
  end subroutine minimize_sgd

end module nf_optimizers
