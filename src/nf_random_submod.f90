submodule(nf_random_mod) nf_random_submod

  ! Provides a random number generator with
  ! normal distribution, centered on zero.

  implicit none

  real(rk), parameter :: pi = 4 * atan(1._rk)

contains

  module procedure randn1d
    real(rk) :: r2(n)
    call random_number(r)
    call random_number(r2)
    r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
  end procedure randn1d

  module procedure randn2d
    real(rk) :: r2(m, n)
    call random_number(r)
    call random_number(r2)
    r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
  end procedure randn2d

end submodule nf_random_submod
