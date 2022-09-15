program sine
  use nf, only: dense, input, network
  implicit none
  type(network) :: net
  real :: x(1), y(1)
  real, parameter :: pi = 4 * atan(1.)
  integer, parameter :: num_iterations = 100000
  integer, parameter :: test_size = 30
  real :: xtest(test_size), ytest(test_size), ypred(test_size)
  integer :: i, n

  print '("Sine training")'
  print '(60("="))'

  net = network([ &
    input(1), &
    dense(5), &
    dense(1) &
  ])

  call net % print_info()

  xtest = [((i - 1) * 2 * pi / test_size, i = 1, test_size)]
  ytest = (sin(xtest) + 1) / 2

  do n = 0, num_iterations

    call random_number(x)
    x = x * 2 * pi
    y = (sin(x) + 1) / 2

    call net % forward(x)
    call net % backward(y)
    call net % update(1.)

    if (mod(n, 10000) == 0) then
      ypred = [(net % predict([xtest(i)]), i = 1, test_size)]
      print '(i0,1x,f9.6)', n, sum((ypred - ytest)**2) / size(ypred)
    end if

  end do

end program sine
