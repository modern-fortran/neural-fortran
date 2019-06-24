program test_network_save
  use mod_network, only: network_type
  implicit none
  type(network_type) :: net1, net2
  integer :: n
  print *, 'Initializing 2 networks with random weights and biases'
  net1 = network_type([768, 30, 10])
  net2 = network_type([768, 30, 10])

  print *, 'Save network 1 into file'
  call net1 % save('test_network.dat')
  call net2 % load('test_network.dat')
  print *, 'Load network 2 from file'
  do n = 1, size(net1 % layers)
    print *, 'Layer ', n, ', weights equal: ',&
      all(net1 % layers(n) % w == net2 % layers(n) % w),&
      ', biases equal:', all(net1 % layers(n) % b == net2 % layers(n) % b)
  end do
  print *, ''

  print *, 'Setting different activation functions for each layer of network 1'
  call net1 % set_activation([character(len=10) :: 'sigmoid', 'tanh', 'gaussian'])
  print *, 'Save network 1 into file'
  call net1 % save('test_network.dat')
  call net2 % load('test_network.dat')
  print *, 'Load network 2 from file'
  do n = 1, size(net1 % layers)
    print *, 'Layer ', n, ', activation functions equal:',&
     associated(net1 % layers(n) % activation, net2 % layers(n) % activation),&
     '(network 1: ', net1 % layers(n) % activation_str, ', network 2: ', net2 % layers(n) % activation_str,')'
  end do  
end program test_network_save
