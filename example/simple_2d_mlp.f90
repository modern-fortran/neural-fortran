program simple
  use nf, only: dense, fc2d, flatten, linear2d, input, network, sgd, relu, tanhf
  implicit none
  type(network) :: net
  real, allocatable :: x(:, :), y(:)
  integer, parameter :: num_iterations = 25
  integer :: n

  print '("Simple")'
  print '(60("="))'

  net = network([ &
    input(4, 5), &
    fc2d(3, 2, activation=relu()), &
    flatten(), &
    dense(4, activation=tanhf()) &
  ])

  call net % print_info()

  allocate(x(4, 5))
  call random_number(x)
  y = [0.123456, 0.246802, 0.9, 0.001]

  do n = 0, num_iterations

    call net % forward(x)
    call net % backward(y)
    call net % update(optimizer=sgd(learning_rate=0.05))

    if (mod(n, 5) == 0) print *, n, net % predict(x)

  end do

end program simple
