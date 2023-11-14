program simple_rnn
  use nf, only: dense, input, network, rnn, sgd
  implicit none
  type(network) :: net
  real, allocatable :: x(:), y(:), p(:)
  integer, parameter :: num_iterations = 1000
  integer :: n, l

  allocate(p(2))

  print '("Simple RNN")'
  print '(60("="))'

  net = network([ &
    input(3), &
    rnn(5), &
    rnn(1) &
  ])

  call net % print_info()

  x = [0.2, 0.4, 0.6]
  y = [0.123456, 0.246802]

  do n = 0, num_iterations

    do l = 1, size(net % layers)
      if (net % layers(l) % name == 'rnn') call net % layers(l) % set_state()
    end do

    if (mod(n, 100) == 0) then
      p(1:1) = net % predict(x)
      p(2:2) = net % predict(x)
      print '(i4,2(3x,f8.6))', n, p

    else

      call net % forward(x)
      call net % backward(y(1:1))
      call net % update(optimizer=sgd(learning_rate=.001))
      call net % forward(x)
      call net % backward(y(2:2))
      call net % update(optimizer=sgd(learning_rate=.001))
    end if

  end do

end program simple_rnn
