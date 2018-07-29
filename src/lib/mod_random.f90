module mod_random

  ! Provides a random number generator with
  ! normal distribution, centered on zero.

  use mod_kinds, only: ik, rk

  implicit none

  private
  public :: randn

  real(rk), parameter :: pi = 4 * atan(1._rk)

  interface randn
    module procedure :: randn1d, randn2d
  end interface randn

contains

  function randn1d(n) result(r)
    ! Generates n random numbers with a normal distribution.
    integer(ik), intent(in) :: n
    real(rk) :: r(n), r2(n)
    call random_number(r)
    call random_number(r2)
    r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
  end function randn1d

  function randn2d(m, n) result(r)
    ! Generates m x n random numbers with a normal distribution.
    integer(ik), intent(in) :: m, n
    real(rk) :: r(m, n), r2(m, n)
    call random_number(r)
    call random_number(r2)
    r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
  end function randn2d

end module mod_random
