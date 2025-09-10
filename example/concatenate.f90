program concatenate
  use nf, only: dense, input, network, sgd
  implicit none

  type(network) :: net1, net2, net3
  real, allocatable :: x1(:), y1(:)
  real, allocatable :: x2(:), y2(:)
  real, allocatable :: x3(:), y3(:)
  integer, parameter :: num_iterations = 500
  integer :: n

  ! Network 1
  net1 = network([ &
    input(3), &
    dense(2) &
  ])

  x1 = [0.2, 0.4, 0.6]
  y1 = [0.123456, 0.246802]

  do n = 1, num_iterations
    call net1 % forward(x1)
    call net1 % backward(y1)
    call net1 % update(optimizer=sgd(learning_rate=1.))
  end do

  print *, "net1 output: ", net1 % predict(x1)

  ! Network 2
  net2 = network([ &
    input(3), &
    dense(3) &
  ])

  x2 = [0.7, 0.5, 0.3]
  y2 = [0.369258, 0.482604, 0.505050]

  do n = 1, num_iterations
    call net2 % forward(x2)
    call net2 % backward(y2)
    call net2 % update(optimizer=sgd(learning_rate=1.))
  end do

  print *, "net2 output: ", net2 % predict(x2)

  ! Network 3
  net3 = network([ &
    input(size(net1 % predict(x1)) + size(net2 % predict(x2))), &
    dense(5) & 
  ])

  x3 = [net1 % predict(x1), net2 % predict(x2)]
  y3 = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555]

  do n = 1, num_iterations
    call net3 % forward(x3)
    call net3 % backward(y3)
    call net3 % update(optimizer=sgd(learning_rate=1.))
  end do

  print *, "net3 output: ", net3 % predict(x3)

end program concatenate