program simple_conv3d
  use nf, only: conv, dense, input, network, sgd
  implicit none
  type(network) :: net
  real :: x(1, 5, 5, 5)
  real :: y(1)
  integer, parameter :: num_iterations = 200
  integer :: n

  print '("Simple conv3d")'
  print '(60("="))'

  net = network([ &
    input(1, 5, 5, 5), &
    conv(filters=2, kernel_width=3, kernel_height=3, kernel_depth=3), &
    dense(1) &
  ])

  call net % print_info()

  call random_number(x)
  y = [0.123456]

  do n = 0, num_iterations

    call net % forward(x)
    call net % backward(y)
    call net % update(optimizer=sgd(learning_rate=1.))

    if (mod(n, 50) == 0) &
      print '(i4,3x,f8.6)', n, net % predict(x)

  end do

end program simple_conv3d
