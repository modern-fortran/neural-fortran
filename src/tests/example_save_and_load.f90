program example_save_and_load

  use mod_network, only: network_type
  implicit none

  type(network_type) :: net1, net2
  real, allocatable :: input(:), output(:)
  integer :: i

  net1 = network_type([3, 5, 2])

  input = [0.2, 0.4, 0.6]
  output = [0.123456, 0.246802]

  ! train network 1
  do i = 1, 500
    call net1 % train(input, output, eta=1.0)
  end do

  ! save network 1 to file
  call net1 % save('my_simple_net.txt')

  ! load network 2 from file 
  !net2 = network_type([3, 5, 2])
  call net2 % load('my_simple_net.txt')
  call net2 % set_activation('sigmoid')

  print *, 'Network 1 output: ', net1 % output(input) 
  print *, 'Network 2 output: ', net2 % output(input)
  print *, 'Outputs match: ', all(net1 % output(input) == net2 % output(input))

end program example_save_and_load
