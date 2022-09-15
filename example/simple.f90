program simple
  use nf, only: dense, input, network
  implicit none
  type(network) :: net
  real, allocatable :: x(:), y(:)
  integer, parameter :: num_iterations = 500
  integer :: n

  print '("Simple")'
  print '(60("="))'

  net = network([ &
    input(3), &
    dense(5), &
    dense(2) &
  ])

  call net % print_info()

  x = [0.2, 0.4, 0.6]
  y = [0.123456, 0.246802]

  do n = 0, num_iterations

    call net % forward(x)
    call net % backward(y)
    call net % update(1.)

    if (mod(n, 50) == 0) &
      print '(i4,2(3x,f8.6))', n, net % predict(x)

  end do

end program simple
