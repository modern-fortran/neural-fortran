module nf_optimizers

  !! This module provides optimizer types to pass to the network constructor.

  implicit none

  private
  public :: sgd

  type :: sgd
    !! Stochastic Gradient Descent optimizer
    real :: learning_rate
    real :: momentum = 0 !TODO
    logical :: nesterov = .false. !TODO
  end type sgd

end module nf_optimizers
