program example_sine
  use mod_kinds, only: ik, rk
  use mod_network, only: network_type
  implicit none
  type(network_type) :: net
  real(rk) :: cumloss, x, y
  real(rk), parameter :: pi = 4 * atan(1._rk)
  integer(ik) :: i
  net = network_type([1, 5, 1])
  cumloss = 0
  do i = 1, 1000000
    call random_number(x)
    y = (sin(x * 2 * pi) + 1) * 0.5
    call net % train([x], [y], eta=10._rk)
    cumloss = cumloss + net % loss([x], [y])
    print *, i, cumloss / i
  end do
end program example_sine
