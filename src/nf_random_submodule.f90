submodule(nf_random) nf_random_submodule
  implicit none

  real, parameter :: pi = 4 * atan(1.d0)

contains

  module function randn1d(n) result(x)
    integer, intent(in) :: n
    real :: x(n)
    real :: u(n), v(n)
    call random_number(u)
    call random_number(v)
    x = sqrt(-2 * log(u)) * cos(2 * pi * v)
  end function randn1d

  module function randn2d(m, n) result(x)
    integer, intent(in) :: m, n
    real :: x(m,n)
    real :: u(m,n), v(m,n)
    call random_number(u)
    call random_number(v)
    x = sqrt(-2 * log(u)) * cos(2 * pi * v)
  end function randn2d

end submodule nf_random_submodule
