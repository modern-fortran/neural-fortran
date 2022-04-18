submodule(mod_random) mod_random_submodule
  implicit none

  real(rk), parameter :: pi = 4 * atan(1._rk)

contains

  module function randn1d(n) result(r)
    integer(ik), intent(in) :: n
    real(rk) :: r(n), r2(n)
    call random_number(r)
    call random_number(r2)
    r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
  end function randn1d

  module function randn2d(m, n) result(r)
    integer(ik), intent(in) :: m, n
    real(rk) :: r(m, n), r2(m, n)
    call random_number(r)
    call random_number(r2)
    r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
  end function randn2d

end submodule mod_random_submodule
