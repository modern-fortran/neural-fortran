module nf_optimizers

  !! This module provides optimizer types to pass to the network constructor.

  implicit none

  private
  public :: optimizer_base_type, sgd

  type, abstract :: optimizer_base_type
    character(:), allocatable :: name
  end type optimizer_base_type

  type, extends(optimizer_base_type) :: sgd
    !! Stochastic Gradient Descent optimizer
    real :: learning_rate
    real :: momentum = 0 !TODO
    logical :: nesterov = .false. !TODO
  end type sgd

end module nf_optimizers
