program example_simple
  use mod_kinds, only: ik, rk
  use mod_network, only: network_type
  implicit none
  type(network_type) :: net
  real(rk), allocatable :: input(:), output(:)
  integer(ik) :: i
  net = network_type([1, 3, 1])
  input = [0.1_rk]
  output = [0.123456789012345_rk]
  do i = 1, 1000
    call net % train(input, output, eta=1._rk)
    print *, 'Iteration: ', i, 'Output:', net % output(input)
  end do
end program example_simple
