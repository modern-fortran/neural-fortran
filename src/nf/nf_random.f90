module nf_random

  !! Provides a random number generator with normal distribution,
  !! centered on zero, and a Fisher-Yates shuffle.

  implicit none

  private
  public :: random_normal, shuffle

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


  subroutine shuffle(x)
    !! Fisher-Yates shuffle.
    real, intent(in out) :: x(:)
      !! Array to shuffle
    integer :: i, j
    real :: r, temp

    do i = size(x), 2, -1
      call random_number(r)
      j = floor(r * i) + 1
      temp = x(i)
      x(i) = x(j)
      x(j) = temp
    end do

  end subroutine shuffle

end module nf_random
