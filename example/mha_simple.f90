program simple
  use nf, only: dense, input, network, sgd, self_attention, flatten
  implicit none
  type(network) :: net
  real, allocatable :: x(:, :), y(:)
  integer, parameter :: num_iterations = 500
  integer :: n

  print '("Simple")'
  print '(60("="))'

  net = network([ &
    input(3, 8), &
    self_attention(4), &
    flatten(), &
    dense(2) &
  ])

  call net % print_info()

  allocate(x(3, 8))
  call random_number(x)

  y = [0.123456, 0.246802]

  do n = 0, num_iterations

    call net % forward(x)
    call net % backward(y)
    call net % update(optimizer=sgd(learning_rate=1.))

    if (mod(n, 50) == 0) &
      print '(i4,2(3x,f8.6))', n, net % predict(x)

  end do

end program simple
