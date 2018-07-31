program example_simple
  use mod_network, only: network_type
  implicit none
  type(network_type) :: net
  real, allocatable :: input(:), output(:)
  integer :: i
  net = network_type([3, 5, 2])
  input = [0.2, 0.4, 0.6]
  output = [0.123456, 0.246802]
  do i = 1, 500
    call net % train(input, output, eta=1.0)
    print *, 'Iteration: ', i, 'Output:', net % output(input)
  end do
end program example_simple
