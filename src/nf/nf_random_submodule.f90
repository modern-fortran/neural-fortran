submodule(nf_random) nf_random_submodule
  implicit none

  real, parameter :: pi = 4 * atan(1.d0)

contains

  module function randn_1d(i) result(x)
    integer, intent(in) :: i
    real :: x(i)
    real :: u(i), v(i)
    call random_number(u)
    call random_number(v)
    x = sqrt(-2 * log(u)) * cos(2 * pi * v)
  end function randn_1d

  module function randn_2d(i, j) result(x)
    integer, intent(in) :: i, j
    real :: x(i,j)
    real :: u(i,j), v(i,j)
    call random_number(u)
    call random_number(v)
    x = sqrt(-2 * log(u)) * cos(2 * pi * v)
  end function randn_2d

  module function randn_4d(i, j, k, l) result(x)
    integer, intent(in) :: i, j, k, l
    real :: x(i,j,k,l)
    real :: u(i,j,k,l), v(i,j,k,l)
    call random_number(u)
    call random_number(v)
    x = sqrt(-2 * log(u)) * cos(2 * pi * v)
  end function randn_4d

end submodule nf_random_submodule
