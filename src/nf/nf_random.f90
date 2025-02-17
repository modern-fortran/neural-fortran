module nf_random

  !! Provides a random number generator with
  !! normal distribution, centered on zero.

  implicit none

  private
  public :: random_normal, random_he, random_xavier

  real, parameter :: pi = 4 * atan(1.d0)

contains

  impure elemental subroutine random_normal(x)
    !! Sample random numbers from a normal distribution using a Box-Muller
    !! formula.
    real, intent(out) :: x
      !! Scalar or array to be filled with random numbers
    real :: u(2)
    call random_number(u)
    u(1) = 1 - u(1)
    x = sqrt(- 2 * log(u(1))) * cos(2 * pi * u(2))
  end subroutine random_normal

  impure elemental subroutine random_he(x, n_prev)
    !! Kaiming weight initialization
    real, intent(in out) :: x
    integer, intent(in) :: n_prev
    call random_number(x)
    x = x * sqrt(2. / n_prev)
  end subroutine random_he

  impure elemental subroutine random_xavier(x, n_prev)
    !! Kaiming weight initialization
    real, intent(in out) :: x
    integer, intent(in) :: n_prev
    real :: lower, upper
    lower = -(1. / sqrt(real(n_prev)))
    upper = 1. / sqrt(real(n_prev))
    call random_number(x)
    x = lower + x * (upper - lower)
  end subroutine random_xavier
end module nf_random
